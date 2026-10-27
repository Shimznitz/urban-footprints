# src/analysis/evaluate_objects.py
"""Object-level evaluation on tiles that have ground-truth building polygons.

A predicted building is correct (true positive) if it overlaps a real building with
IoU >= 0.5. Run from the project root:

    python -m src.analysis.evaluate_objects --tiles data/raw/sample_tiles --outdir data/outputs_v2/eval
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

from src.postprocess.geo import WGS84, reproject_geom


def load_truth(path, metric_crs):
    """Ground-truth polygons in metres. Empty placeholder features are skipped."""
    polys = []
    for f in json.load(open(path)).get("features", []):
        g = f.get("geometry")
        if not g or not g.get("coordinates"):
            continue
        try:
            geom = shape(g)
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)                      # repair self-intersections
        polys.append(reproject_geom(geom, WGS84, metric_crs))
    return polys


def match(pred, truth, iou_thr=0.5):
    """One-to-one matching by IoU. Returns (tp, fp, fn)."""
    if not pred or not truth:
        return 0, len(pred), len(truth)
    tree = STRtree(truth)
    pairs = []
    for i, p in enumerate(pred):
        for j in tree.query(p, predicate="intersects"):
            inter = p.intersection(truth[int(j)]).area
            union = p.area + truth[int(j)].area - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                pairs.append((iou, i, int(j)))
    pairs.sort(reverse=True)                           # best overlaps first
    used_p, used_t, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/raw/sample_tiles",
                    help="folder with 8-band .tif tiles and a geojson/ subfolder")
    ap.add_argument("--ckpt", default="models/unet_resnet50_8band.pth")
    ap.add_argument("--stats", default="models/band_stats.json")
    ap.add_argument("--outdir", default="data/outputs_v2/eval")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=15.0)
    ap.add_argument("--split-h", type=float, default=0.6)
    a = ap.parse_args()

    import rasterio
    from src.model.predict import load_model, predict_tif
    from src.postprocess.footprints import pixel_size_m, prob_to_footprints
    from src.postprocess.geo import metric_crs_for

    model = load_model(a.ckpt)
    stats = json.load(open(a.stats))
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for tif in sorted(glob.glob(f"{a.tiles}/*.tif")):
        tid = os.path.basename(tif)[:-4].split("_")[-1]            # e.g. img1040
        gj = f"{a.tiles}/geojson/buildings_AOI_5_Khartoum_{tid}.geojson"
        if not os.path.exists(gj):
            print("skip (no ground truth):", tid)
            continue
        t0 = time.time()
        _, prob, _, prof = predict_tif(model, tif, stats)
        with rasterio.open(tif) as s:
            px_m, bounds = float(np.mean(pixel_size_m(s))), s.bounds
        metric = metric_crs_for(prof["crs"], bounds)
        gm, gll = prob_to_footprints(prob, prof["transform"], prof["crs"], px_m, metric,
                                     thr=a.thr, min_area_m2=a.min_area, split_h_m=a.split_h)
        truth = load_truth(gj, metric)
        pred = list(gm.geometry)
        tp, fp, fn = match(pred, truth)
        p, r, f = prf(tp, fp, fn)
        rows.append(dict(tile=tid, truth=len(truth), pred=len(pred), TP=tp, FP=fp, FN=fn,
                         precision=round(p, 3), recall=round(r, 3), F1=round(f, 3),
                         truth_m2=round(sum(g.area for g in truth)),
                         pred_m2=round(float(gm.area_m2.sum())) if len(gm) else 0,
                         truth_tiny=sum(g.area < a.min_area for g in truth)))
        if len(gll):
            warnings.filterwarnings("ignore", message=".crs. was not provided")
            gll.to_file(f"{a.outdir}/footprints_{tid}.geojson", driver="GeoJSON")
        print(f"{tid}: {time.time() - t0:.0f}s | truth {len(truth)} | pred {len(pred)} | F1 {f:.3f}")

    if not rows:
        raise SystemExit("no tiles with ground truth found")
    df = pd.DataFrame(rows)
    tp, fp, fn = int(df.TP.sum()), int(df.FP.sum()), int(df.FN.sum())
    p, r, f = prf(tp, fp, fn)
    total = dict(tile="TOTAL", truth=int(df.truth.sum()), pred=int(df.pred.sum()), TP=tp, FP=fp,
                 FN=fn, precision=round(p, 3), recall=round(r, 3), F1=round(f, 3),
                 truth_m2=int(df.truth_m2.sum()), pred_m2=int(df.pred_m2.sum()),
                 truth_tiny=int(df.truth_tiny.sum()))
    out = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    out.to_csv(f"{a.outdir}/evaluation_objects.csv", index=False)
    print()
    print(out.to_string(index=False))
    print(f"\npredicted/real area: {total['pred_m2'] / max(total['truth_m2'], 1):.2f} | "
          f"predicted/real count: {total['pred'] / max(total['truth'], 1):.2f}")


if __name__ == "__main__":
    main()