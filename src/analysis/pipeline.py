# src/analysis/pipeline.py
"""Analyse one uploaded GeoTIFF: check it is compatible, cut it into tile-sized blocks,
run model + footprints + indicators on each block."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box

from src.analysis.indicators import tile_indicators
from src.postprocess.footprints import pixel_size_m, prob_to_footprints
from src.postprocess.geo import WGS84, metric_crs_for, reproject_geom

BLOCK = 650        # pixels per block: the tile size the model was validated on
MIN_BLOCK = 128    # skip leftover strips narrower than this
MAX_BLOCKS = 16    # keeps CPU analysis to a few minutes
RGB = (4, 2, 1)    # red, green, blue positions in the 8-band order


def inspect_raster(path):
    """Return (errors, warnings, info). Errors mean the image cannot be analysed."""
    errors, warns = [], []
    with rasterio.open(path) as s:
        info = dict(bands=s.count, width=s.width, height=s.height,
                    crs=str(s.crs) if s.crs else None, dtype=s.dtypes[0])
        if s.count != 8:
            errors.append(f"The model needs an 8-band multispectral image (WorldView-3 band order); "
                          f"this file has {s.count} band(s).")
        if s.crs is None:
            errors.append("The image has no georeferencing (CRS), so areas and map positions "
                          "cannot be computed.")
        if errors:
            return errors, warns, info

        px_x, px_y = pixel_size_m(s)
        px = float(np.mean((px_x, px_y)))
        info["pixel_m"] = round(px, 2)
        if px > 0.7:
            errors.append(f"Pixel size is about {px:.2f} m. The model needs roughly 0.3 to 0.6 m "
                          f"imagery to separate buildings; coarser imagery (for example Sentinel-2 "
                          f"at 10 m) cannot work.")
        elif px < 0.15:
            warns.append(f"Pixel size is unusually fine ({px:.2f} m); results may differ from "
                         f"what the model was validated on.")

        sample = s.read(out_shape=(8, min(s.height, 256), min(s.width, 256)))
        nz = sample[sample > 0]
        info["p99"] = float(np.percentile(nz, 99)) if nz.size else 0.0
        if info["p99"] < 150 or info["p99"] > 10000:
            warns.append("Pixel values look different from the training data (roughly 11-bit digital "
                         "numbers, in the hundreds to low thousands). If the image is 8-bit, "
                         "reflectance-scaled or from another sensor, results may be unreliable.")

        info["blocks"] = math.ceil(s.height / BLOCK) * math.ceil(s.width / BLOCK)
        if info["blocks"] > MAX_BLOCKS:
            errors.append(f"The image needs {info['blocks']} tile-sized blocks; this app handles up "
                          f"to {MAX_BLOCKS} (about {MAX_BLOCKS * 3.5:.0f} hectares). Please crop it.")
    return errors, warns, info


def rgb_image(arr, stats):
    """uint8 RGB preview with the same fixed stretch as the demo thumbnails."""
    mean, std = np.array(stats["mean"], "float32"), np.array(stats["std"], "float32")
    chans = []
    for b in RGB:
        lo, hi = mean[b] - 2 * std[b], mean[b] + 2 * std[b]
        chans.append(np.clip((arr[b].astype("float32") - lo) / (hi - lo), 0, 1))
    return (np.dstack(chans) * 255).astype("uint8")


def analyze_array(model, arr, transform, crs, stats, device="cpu", thr=0.5,
                  min_area_m2=15.0, split_h_m=0.6):
    """One block: model -> footprints -> indicators. Returns (row, footprints in lon/lat)."""
    from src.model.predict import predict_array          # imported here: needs torch

    mean, std = np.array(stats["mean"], "float32"), np.array(stats["std"], "float32")
    prob, valid = predict_array(model, arr, mean, std, device=device)
    h, w = arr.shape[1:]
    west, south, east, north = array_bounds(h, w, transform)
    bounds = SimpleNamespace(left=west, right=east, bottom=south, top=north)
    px_x, px_y = pixel_size_m(SimpleNamespace(transform=transform, crs=crs, bounds=bounds))
    metric = metric_crs_for(crs, bounds)
    gm, gll = prob_to_footprints(prob, transform, crs, float(np.mean((px_x, px_y))), metric,
                                 thr=thr, min_area_m2=min_area_m2, split_h_m=split_h_m)
    w_, s_, e_, n_ = reproject_geom(box(west, south, east, north), crs, WGS84).bounds
    row = dict(west=w_, south=s_, east=e_, north=n_, **tile_indicators(gm, arr, valid, px_x * px_y))
    return row, gll


def analyze_file(path, model, stats, device="cpu", progress=None, prefix="", **kw):
    """Whole image -> dict(df, features, thumbs). progress(k, n) is called after each block.
    prefix is added to every block id (use it to keep ids unique across several files)."""
    rows, features, thumbs = [], [], {}
    with rasterio.open(path) as s:
        H, W, crs = s.height, s.width, s.crs
        wins = [Window(c, r, min(BLOCK, W - c), min(BLOCK, H - r))
                for r in range(0, H, BLOCK) for c in range(0, W, BLOCK)]
        wins = [w for w in wins if w.width >= MIN_BLOCK and w.height >= MIN_BLOCK]
        for k, win in enumerate(wins, 1):
            arr = s.read(window=win)
            tid = f"{prefix}block_{k}"
            row, gll = analyze_array(model, arr, s.window_transform(win), crs, stats, device, **kw)
            row["tile"] = tid
            rows.append(row)
            for geom, props in zip(gll.geometry, gll.drop(columns="geometry").to_dict("records")):
                props["tile"] = tid
                features.append({"type": "Feature", "properties": props,
                                 "geometry": geom.__geo_interface__})
            thumbs[tid] = rgb_image(arr, stats)
            if progress:
                progress(k, len(wins))
    return dict(df=pd.DataFrame(rows), features=features, thumbs=thumbs)