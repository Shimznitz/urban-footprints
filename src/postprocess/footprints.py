# src/postprocess/footprints.py
import argparse
import os
import warnings

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import shape
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from src.postprocess.geo import WGS84, metric_crs_for, reproject_geom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pixel_size_m(src):
    """Approximate pixel width and height in metres."""
    t = src.transform
    if src.crs is not None and src.crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2
        return abs(t.a) * 111_320 * np.cos(np.radians(lat)), abs(t.e) * 110_540
    factor = src.crs.linear_units_factor[1] if src.crs is not None else 1.0
    return abs(t.a) * factor, abs(t.e) * factor


def fill_small_holes(mask, max_px):
    """Fill background holes smaller than max_px pixels (keeps real courtyards)."""
    holes = ndi.binary_fill_holes(mask) & ~mask
    lab, n = ndi.label(holes)
    if n == 0:
        return mask
    sizes = ndi.sum(holes, lab, range(1, n + 1))
    small = np.isin(lab, np.where(sizes < max_px)[0] + 1)
    return mask | small


def split_touching(mask, px_m, h_m=0.6, min_half_width_m=1.0):
    """Distance transform + watershed: cut blobs where a narrow neck joins two peaks."""
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(mask), 1.0)
    peaks = h_maxima(dist, h_m / px_m).astype(bool) & (dist >= min_half_width_m / px_m)
    markers, _ = ndi.label(peaks)
    labels = watershed(-dist, markers, mask=mask)
    orphan, _ = ndi.label(mask & (labels == 0))      # thin shapes that got no peak
    labels[orphan > 0] = orphan[orphan > 0] + labels.max()
    return labels


def prob_to_footprints(prob, transform, crs, px_m, metric_crs, thr=0.5, min_area_m2=15.0,
                       split=True, split_h_m=0.6, simplify_m=0.4):
    """Probability map -> (gm, gll): building polygons in metres and in lon/lat.

    Neither GeoDataFrame carries a CRS object (that would need pyproj). gm holds
    coordinates in metric_crs (metres); gll holds WGS84 lon/lat, ready for web maps.
    """
    mask = fill_small_holes(prob > thr, max_px=int(4.0 / px_m ** 2))
    labels = split_touching(mask, px_m, split_h_m) if split else ndi.label(mask)[0]
    labels = labels.astype("int32")

    polys, vals = [], []
    for geom, v in shapes(labels, mask=labels > 0, transform=transform):
        polys.append(reproject_geom(shape(geom), crs, metric_crs))   # -> metres
        vals.append(int(v))

    cols = ["building_id", "mean_prob", "area_m2", "perimeter_m", "compactness"]
    if not polys:
        empty = gpd.GeoDataFrame({c: [] for c in cols}, geometry=[])
        return empty, empty.copy()

    ids = np.unique(labels[labels > 0])
    mean_prob = dict(zip(ids, ndi.mean(prob, labels, ids)))          # model confidence
    gm = gpd.GeoDataFrame(
        {"mean_prob": [float(mean_prob[v]) for v in vals]},
        geometry=[p.simplify(simplify_m, preserve_topology=True) for p in polys],
    )
    gm["area_m2"] = gm.geometry.area
    gm = gm[gm.area_m2 >= min_area_m2].copy()
    gm["perimeter_m"] = gm.geometry.length
    gm["compactness"] = 4 * np.pi * gm.area_m2 / gm.perimeter_m ** 2
    gm.insert(0, "building_id", range(1, len(gm) + 1))

    gll = gm.copy()
    gll["geometry"] = [reproject_geom(g, metric_crs, WGS84) for g in gm.geometry]
    return gm, gll


def stretch(b):
    lo, hi = np.percentile(b, (2, 98))
    return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prob_tif", help="probability GeoTIFF from src.model.predict")
    ap.add_argument("--image", help="original 8-band tif, for the overlay picture")
    ap.add_argument("--outdir", default="data/outputs")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=15.0, help="smallest building, m2")
    ap.add_argument("--split-h", type=float, default=0.6,
                    help="neck depth in metres needed to split; higher = fewer splits")
    ap.add_argument("--no-split", action="store_true")
    a = ap.parse_args()

    with rasterio.open(a.prob_tif) as src:
        prob, transform, crs, bounds = src.read(1), src.transform, src.crs, src.bounds
        px_m = float(np.mean(pixel_size_m(src)))
    metric = metric_crs_for(crs, bounds)
    print(f"pixel size ~{px_m:.2f} m | working CRS: EPSG:{metric.to_epsg()}")

    gm, gll = prob_to_footprints(prob, transform, crs, px_m, metric, thr=a.thr,
                                 min_area_m2=a.min_area, split=not a.no_split,
                                 split_h_m=a.split_h)
    if gm.empty:
        raise SystemExit("no buildings found at this threshold")

    base = os.path.basename(a.prob_tif)[:-4]
    name = base[len("prob_"):] if base.startswith("prob_") else base
    os.makedirs(a.outdir, exist_ok=True)
    out_geo = f"{a.outdir}/footprints_{name}.geojson"
    warnings.filterwarnings("ignore", message=".crs. was not provided")
    gll.to_file(out_geo, driver="GeoJSON")      # lon/lat; GeoJSON is WGS84 by definition
    print(f"buildings: {len(gm)} | total footprint: {gm.area_m2.sum():.0f} m2 | "
          f"median: {gm.area_m2.median():.0f} m2 | mean confidence: {gm.mean_prob.mean():.2f}")
    print("saved:", out_geo)

    if a.image:
        with rasterio.open(a.image) as s:
            arr, b = s.read(), s.bounds
        rgb = np.dstack([stretch(arr[i]) for i in (4, 2, 1)])
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(rgb, extent=(b.left, b.right, b.bottom, b.top))
        gll.boundary.plot(ax=ax, color="red", linewidth=0.7)
        ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)
        ax.set_title(f"{len(gm)} buildings (threshold {a.thr})"); ax.axis("off")
        out_png = f"{a.outdir}/overlay_{name}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print("saved:", out_png)