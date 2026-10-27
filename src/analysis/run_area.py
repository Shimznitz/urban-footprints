# src/analysis/run_area.py
"""Run the whole pipeline over a folder of 8-band tiles and write what the dashboard needs.

    python -m src.analysis.run_area --tiles data/raw/area_tiles --outdir data/outputs_v2/area

Outputs in --outdir:
    tile_indicators.csv     one row per tile (bounds + indicators, + ground truth if available)
    footprints_all.geojson  every building polygon (lon/lat), tagged with its tile
    thumbs/<tile>.jpg       colour thumbnail of each tile for the map
    per_tile/               cached per-tile results (reruns skip finished tiles)
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from shapely.geometry import box

from src.analysis.evaluate_objects import load_truth
from src.analysis.indicators import tile_indicators
from src.postprocess.geo import WGS84, metric_crs_for, reproject_geom

RGB = (4, 2, 1)   # red, green, blue positions in the 8-band order


def thumbnail(arr, stats, path):
    """Colour thumbnail with the SAME stretch for every tile, so the mosaic looks consistent."""
    from PIL import Image
    mean = np.array(stats["mean"], "float32")
    std = np.array(stats["std"], "float32")
    chans = []
    for b in RGB:
        lo, hi = mean[b] - 2 * std[b], mean[b] + 2 * std[b]
        chans.append(np.clip((arr[b].astype("float32") - lo) / (hi - lo), 0, 1))
    Image.fromarray((np.dstack(chans) * 255).astype("uint8")).save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/raw/area_tiles")
    ap.add_argument("--outdir", default="data/outputs_v2/area")
    ap.add_argument("--ckpt", default="models/unet_resnet50_8band.pth")
    ap.add_argument("--stats", default="models/band_stats.json")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=15.0)
    ap.add_argument("--split-h", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tiles (quick test)")
    ap.add_argument("--force", action="store_true", help="recompute tiles already done")
    a = ap.parse_args()

    import rasterio
    from src.model.predict import load_model, predict_tif
    from src.postprocess.footprints import pixel_size_m, prob_to_footprints

    os.makedirs(f"{a.outdir}/per_tile", exist_ok=True)
    os.makedirs(f"{a.outdir}/thumbs", exist_ok=True)
    stats = json.load(open(a.stats))
    tifs = sorted(glob.glob(f"{a.tiles}/*.tif"))
    if a.limit:
        tifs = tifs[:a.limit]
    if not tifs:
        raise SystemExit(f"no .tif files found in {a.tiles}")

    model, rows, features, t_start = None, [], [], time.time()
    for k, tif in enumerate(tifs, 1):
        tid = os.path.basename(tif)[:-4].split("_")[-1]                 # e.g. img1040
        f_ind, f_geo = f"{a.outdir}/per_tile/{tid}.json", f"{a.outdir}/per_tile/{tid}.geojson"
        f_thumb = f"{a.outdir}/thumbs/{tid}.jpg"
        if not a.force and all(os.path.exists(p) for p in (f_ind, f_geo, f_thumb)):
            row = json.load(open(f_ind))
            feats = json.load(open(f_geo))["features"]
            print(f"[{k}/{len(tifs)}] {tid}: cached")
        else:
            if model is None:
                model = load_model(a.ckpt)
            t0 = time.time()
            arr, prob, valid, prof = predict_tif(model, tif, stats)
            with rasterio.open(tif) as s:
                px_x, px_y = pixel_size_m(s)
                bounds = s.bounds
            metric = metric_crs_for(prof["crs"], bounds)
            gm, gll = prob_to_footprints(prob, prof["transform"], prof["crs"],
                                         float(np.mean((px_x, px_y))), metric, thr=a.thr,
                                         min_area_m2=a.min_area, split_h_m=a.split_h)
            west, south, east, north = reproject_geom(box(*bounds), prof["crs"], WGS84).bounds
            row = dict(tile=tid, west=west, south=south, east=east, north=north,
                       **tile_indicators(gm, arr, valid, px_x * px_y))

            gj = f"{a.tiles}/geojson/buildings_AOI_5_Khartoum_{tid}.geojson"
            if os.path.exists(gj):                                      # ground truth, if present
                truth = load_truth(gj, metric)
                row["truth_n"] = len(truth)
                row["truth_m2"] = int(round(sum(g.area for g in truth)))

            if len(gll):
                warnings.filterwarnings("ignore", message=".crs. was not provided")
                gll.to_file(f_geo, driver="GeoJSON")
            else:
                json.dump({"type": "FeatureCollection", "features": []}, open(f_geo, "w"))
            feats = json.load(open(f_geo))["features"]
            json.dump(row, open(f_ind, "w"))
            thumbnail(arr, stats, f_thumb)
            print(f"[{k}/{len(tifs)}] {tid}: {row['n_buildings']} buildings | "
                  f"built-up {row['built_frac']:.0%} | {time.time() - t0:.0f}s")

        for f in feats:
            f["properties"]["tile"] = tid
        features.extend(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    man = f"{a.tiles}/area_manifest.csv"
    if os.path.exists(man):
        df = df.merge(pd.read_csv(man)[["tile", "split"]], on="tile", how="left")
    df.to_csv(f"{a.outdir}/tile_indicators.csv", index=False)
    json.dump({"type": "FeatureCollection", "features": features},
              open(f"{a.outdir}/footprints_all.geojson", "w"))
    print(f"\ndone: {len(df)} tiles | {int(df.n_buildings.sum())} buildings | "
          f"{df.built_m2.sum() / 1e4:.1f} ha built-up | {(time.time() - t_start) / 60:.1f} min")
    print("saved:", f"{a.outdir}/tile_indicators.csv", "and footprints_all.geojson")


if __name__ == "__main__":
    main()