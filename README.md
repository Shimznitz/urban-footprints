# README.md
# Urban Footprints

Building footprints from multispectral satellite imagery, turned into a plain-language picture of
building cover, crowding and green space for urban planners.
AI4SDGs 2026 · Track 1 (Building Mapping for Sustainable Cities, SDG 11).

**Live demo (no install):** <HOSTED-DEMO-LINK>

## What it does
1. A U-Net with a ResNet-50 encoder reads all 8 spectral bands and outlines every building.
2. We measure building cover, crowding, building size and greenery for each ~190 m block.
3. Blocks are ranked with a transparent score; plain-language findings and suggested actions follow.
4. A one-page planning brief can be downloaded and printed.

## Try it in 5 minutes (no GPU, no model files)
```bash
git clone https://github.com/Shimznitz/urban-footprints.git
cd urban-footprints
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash. macOS/Linux: source .venv/bin/activate
                                  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Check that everything works
```bash
python -m src.selfcheck
```
It prints OK, FAIL or SKIP for each part. SKIP means an optional part (the model) is not installed.

## Analyse your own imagery (upload mode)
```bash
pip install -r requirements-model.txt
```
Download `unet_resnet50_8band.pth` from (https://github.com/Shimznitz/urban-footprints/releases/tag/v0.1) into `models/`, then open the dashboard,
choose **Analyse your own imagery** and drag in one or more GeoTIFFs.

The imagery must be: **8 bands in WorldView-3 band order, about 0.3 to 0.6 m pixels, georeferenced,
11-bit-style digital numbers.** The app refuses anything else with an explanation.
Sample tiles are in `data/sample/`.

Command line, for any image:
```bash
python -m src.model.predict data/sample/MUL-PanSharpen_AOI_5_Khartoum_img1028.tif --outdir out
python -m src.postprocess.footprints out/prob_MUL-PanSharpen_AOI_5_Khartoum_img1028.tif \
    --image data/sample/MUL-PanSharpen_AOI_5_Khartoum_img1028.tif --outdir out
```

## Using imagery from another city
Drop the GeoTIFFs into upload mode (choose "rank against: only the blocks in my upload"), or process a
folder and view it:
```bash
python -m src.analysis.run_area --tiles data/raw/my_city --outdir data/demo_my_city
URBAN_DEMO_DIR=data/demo_my_city streamlit run dashboard/app.py
```
The model was trained on one city (Khartoum). Results elsewhere are untested; fine-tuning on a few
hundred labelled tiles from the new city is the recommended next step.

## Reproduce our accuracy numbers
```bash
python -m src.analysis.evaluate_objects --tiles data/sample
```
Needs the weights. Prints per-tile and total precision, recall and F1 for the 8 held-out test tiles.

| Measure (Khartoum, held-out tiles) | Result |
|---|---|
| Pixel-level overlap (IoU) | 0.69 |
| Object-level F1 (IoU >= 0.5), 8 test tiles | about 0.60 (precision 0.61, recall 0.59; rough 95% interval 0.55 to 0.66) |
| Built-up area per tile vs real | about -9% to +16% |
| Building count per tile vs real | about -14% to +34% (counts are approximate) |

## Training
Trained on Kaggle's GPU on SpaceNet 2 (Khartoum, 8-band WorldView-3). See `notebooks/` and
`src/data/dataset.py`, `src/model/loss.py`, `src/model/train.py`. U-Net, ImageNet-pretrained ResNet-50
encoder with 8 input channels, loss = 0.5 BCE + 0.5 Dice (empty pixels masked), AdamW, 3 epochs decoder
only then up to 20 epochs full, random 256 px crops with flips and rotations. Train/validation/test are
split by city area (711 / 211 / 90 tiles, seed 42), not at random.

## Known limits
- Roofs that look like bare ground (large flat roofs, dark roofs) are under-detected; touching
  buildings can merge; outlines are smooth blobs, not exact rectangles.
- One city, one sensor. Vegetation uses an uncalibrated NDVI threshold (0.2).
- The score ranks blocks against a reference area. It is a screening aid built from image proxies, not
  proof that an area is an informal settlement. Validate on the ground.

## Layout
```
dashboard/app.py          the dashboard
src/analysis/             indicators, scoring, plain-language insights, batch and upload pipelines
src/model/                prediction, loss, training
src/postprocess/          threshold, watershed split, polygons, coordinate helpers
src/selfcheck.py          health check
data/demo/                precomputed Khartoum demo results
data/sample/              8 held-out test tiles with labels
models/band_stats.json    per-band normalisation used in training
```

## Data and credits
SpaceNet 2 (Khartoum), CC BY-SA 4.0. Basemaps: OpenStreetMap contributors, Esri World Imagery.

## Team
CTPlanners