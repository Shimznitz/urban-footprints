# src/model/predict.py
import argparse
import json
import os

import numpy as np
import rasterio
import torch

from src.model.train import build_model


def load_model(ckpt_path, device="cpu"):
    model = build_model(in_channels=8, pretrained=False)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()

def _window_weight(crop):
    """Centre-weighted blending: a pixel near a window's edge counts for less."""
    w1 = np.hanning(crop + 2)[1:-1].astype("float32")   # smooth bell, never exactly 0
    return np.maximum(np.outer(w1, w1), 0.01)

def _starts(size, crop, stride):
    """Window start positions that cover the whole axis, including the last edge."""
    s = list(range(0, max(size - crop, 0) + 1, stride))
    if s[-1] + crop < size:
        s.append(size - crop)
    return s


@torch.no_grad()
def predict_array(model, arr, mean, std, crop=256, stride=128, batch=8, device="cpu"):
    """arr: (8, H, W) raw pixel values -> (prob (H, W) in 0..1, valid (H, W) bool)."""
    bands, H, W = arr.shape
    valid = arr.max(axis=0) > 0                      # all bands zero = empty space
    x = (arr.astype("float32") - mean[:, None, None]) / std[:, None, None]
    x = x * valid                                    # same preprocessing as training

    ph, pw = max(crop - H, 0), max(crop - W, 0)      # pad only if the image is tiny
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
    Hp, Wp = x.shape[1:]

    acc = np.zeros((Hp, Wp), dtype="float32")        # sum of predictions
    cnt = np.zeros((Hp, Wp), dtype="float32")        # how many windows covered each pixel
    coords = [(y, xx) for y in _starts(Hp, crop, stride) for xx in _starts(Wp, crop, stride)]
    w = _window_weight(crop)

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, y:y + crop, xx:xx + crop] for y, xx in chunk]))
        pb = torch.sigmoid(model(xb.to(device)).float()).cpu().numpy()[:, 0]
        for (y, xx), p in zip(chunk, pb):
            acc[y:y + crop, xx:xx + crop] += p * w
            cnt[y:y + crop, xx:xx + crop] += w

    prob = (acc / np.maximum(cnt, 1e-6))[:H, :W]
    return prob * valid, valid                       # remove phantom buildings in empty space


def predict_tif(model, tif_path, stats, **kw):
    with rasterio.open(tif_path) as src:
        arr = src.read()
        profile = src.profile
    mean, std = np.array(stats["mean"], "float32"), np.array(stats["std"], "float32")
    prob, valid = predict_array(model, arr, mean, std, **kw)
    return arr, prob, valid, profile


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("--ckpt", default="models/unet_resnet50_8band.pth")
    ap.add_argument("--stats", default="models/band_stats.json")
    ap.add_argument("--outdir", default="data/outputs")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.outdir, exist_ok=True)
    model = load_model(a.ckpt)
    stats = json.load(open(a.stats))
    arr, prob, valid, profile = predict_tif(model, a.tif, stats)

    name = os.path.splitext(os.path.basename(a.tif))[0]
    profile.update(count=1, dtype="float32", compress="lzw")
    with rasterio.open(f"{a.outdir}/prob_{name}.tif", "w", **profile) as dst:
        dst.write(prob.astype("float32")[None])      # georeferenced probability map

    def stretch(b):
        lo, hi = np.percentile(b, (2, 98))
        return np.clip((b - lo) / (hi - lo + 1e-6), 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(np.dstack([stretch(arr[b]) for b in (4, 2, 1)])); ax[0].set_title("image")
    ax[1].imshow(prob, cmap="magma", vmin=0, vmax=1); ax[1].set_title("building probability")
    for x in ax:
        x.axis("off")
    plt.savefig(f"{a.outdir}/preview_{name}.png", dpi=120, bbox_inches="tight")
    print("saved:", f"{a.outdir}/prob_{name}.tif", "and preview_" + name + ".png")