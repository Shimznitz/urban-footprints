# src/analysis/indicators.py
"""Per-tile urban indicators from the building footprints and the 8-band image."""
import numpy as np
from scipy.spatial import cKDTree

# 0-based band positions in the 8-band WorldView-3 order:
# coastal, blue, green, yellow, red, red edge, NIR1, NIR2
RED, NIR1 = 4, 6


def ndvi(arr):
    """Vegetation index (NIR - red) / (NIR + red). Pixel values are uncalibrated, so treat as relative."""
    red = arr[RED].astype("float32")
    nir = arr[NIR1].astype("float32")
    return (nir - red) / np.maximum(nir + red, 1e-6)


def tile_indicators(gm, arr, valid, px_area_m2, veg_thr=0.2, small_m2=50.0):
    """gm: building table in metres (from prob_to_footprints); arr: (8, H, W); valid: (H, W) bool."""
    valid_px = int(valid.sum())
    valid_m2 = valid_px * px_area_m2                  # real imagery only, empty space excluded
    n = len(gm)
    areas = gm["area_m2"].to_numpy() if n else np.array([])
    built = float(areas.sum())

    out = dict(
        valid_m2=int(round(valid_m2)),
        n_buildings=int(n),
        built_m2=int(round(built)),
        built_frac=built / valid_m2 if valid_m2 > 0 else float("nan"),
        density_per_ha=n / (valid_m2 / 1e4) if valid_m2 > 0 else float("nan"),
        median_area_m2=float(np.median(areas)) if n else float("nan"),
        small_share=float((areas < small_m2).mean()) if n else float("nan"),
        mean_conf=float(gm["mean_prob"].mean()) if n else float("nan"),
    )
    if n >= 2:                                        # crowding: gap to the nearest neighbour
        pts = np.array([(g.centroid.x, g.centroid.y) for g in gm.geometry])
        d, _ = cKDTree(pts).query(pts, k=2)
        out["nn_dist_m"] = float(np.median(d[:, 1]))
    else:
        out["nn_dist_m"] = float("nan")

    veg = (ndvi(arr) > veg_thr) & valid
    out["veg_frac"] = float(veg.sum() / max(valid_px, 1))
    return out