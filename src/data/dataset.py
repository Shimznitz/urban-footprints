# src/data/dataset.py
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

TILE = 650  # every SpaceNet 2 tile is 650 x 650 pixels


class KhartoumTiles(Dataset):
    """Serves (image, mask, valid) crops from the Khartoum tiles.

    image: (8, crop, crop) float32, standardized per band, empty pixels set to 0
    mask : (1, crop, crop) float32, 1 = building
    valid: (1, crop, crop) float32, 1 = real pixel, 0 = empty space (ignored by loss/metrics)
    """

    def __init__(self, img_dir, prep_dir, split, mean, std,
                 crop=256, train=True, crops_per_tile=4, min_valid=0.5):
        self.img_dir, self.prep_dir = img_dir, prep_dir
        self.mean = np.asarray(mean, dtype="float32")[:, None, None]
        self.std = np.asarray(std, dtype="float32")[:, None, None]
        self.crop, self.train, self.min_valid = crop, train, min_valid

        manifest = pd.read_csv(f"{prep_dir}/manifest.csv")
        tiles = manifest.loc[manifest["split"] == split, "tile"].tolist()

        if train:
            # random crops: each tile appears several times, a new random crop each time
            self.items = [(t, None, None) for t in tiles for _ in range(crops_per_tile)]
        else:
            # fixed 3x3 grid of slightly overlapping crops, so evaluation is repeatable
            starts = np.linspace(0, TILE - crop, 3).astype(int)
            self.items = []
            for t in tiles:
                valid = np.load(f"{prep_dir}/masks/{t}.npz")["valid"]
                for y in starts:
                    for x in starts:
                        if valid[y:y + crop, x:x + crop].mean() >= min_valid:
                            self.items.append((t, int(x), int(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tile, x, y = self.items[i]
        c = self.crop
        z = np.load(f"{self.prep_dir}/masks/{tile}.npz")
        mask_full, valid_full = z["mask"], z["valid"]
        rng = np.random.default_rng()  # fresh randomness (safe with several loader workers)

        if x is None:  # training: pick a random crop that holds enough real pixels
            for _ in range(10):
                x, y = (int(v) for v in rng.integers(0, TILE - c + 1, size=2))
                if valid_full[y:y + c, x:x + c].mean() >= self.min_valid:
                    break

        # read only the crop from disk, not the whole tile
        with rasterio.open(f"{self.img_dir}/MUL-PanSharpen_AOI_5_Khartoum_{tile}.tif") as src:
            img = src.read(window=Window(x, y, c, c)).astype("float32")  # (8, c, c)
        m = mask_full[y:y + c, x:x + c].astype("float32")
        v = valid_full[y:y + c, x:x + c].astype("float32")

        img = (img - self.mean) / self.std   # standardize each band
        img = img * v                        # empty pixels -> 0 (the average value)

        if self.train:                       # augmentation = regularization
            k = int(rng.integers(4))         # rotate by 0/90/180/270 degrees
            img, m, v = (np.rot90(a, k, axes=(-2, -1)) for a in (img, m, v))
            if rng.random() < 0.5:           # horizontal flip
                img, m, v = (np.flip(a, axis=-1) for a in (img, m, v))
            gain = 1 + rng.uniform(-0.1, 0.1)   # brightness/contrast jitter, image only
            offset = rng.uniform(-0.1, 0.1)
            img = (img * gain + offset) * v

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype="float32"))
        return to_t(img), to_t(m)[None], to_t(v)[None]