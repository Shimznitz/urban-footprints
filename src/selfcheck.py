# src/selfcheck.py
"""One-command health check for the project.   Run from the project root:

    python -m src.selfcheck

It tells you what works on this machine: the dashboard libraries, the bundled demo data, the
scoring and briefing code, and (if the model files are present) the full model pipeline on a
bundled sample tile. Missing optional parts are reported as SKIP, not as failures.
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHTS = ROOT / "models" / "unet_resnet50_8band.pth"
STATS = ROOT / "models" / "band_stats.json"
SAMPLE_DIR = ROOT / "data" / "sample"
# what the reference setup produced for sample tile img1028 (a regression check, with tolerance)
EXPECT_N, EXPECT_M2 = 21, 3656

results = {"ok": 0, "fail": 0, "skip": 0}


def run(name, fn):
    try:
        msg = fn()
        print(f"[ OK ] {name}" + (f": {msg}" if msg else ""))
        results["ok"] += 1
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        results["fail"] += 1
        return False


def skip(name, why):
    print(f"[SKIP] {name}: {why}")
    results["skip"] += 1


def versions(mods):
    out = []
    for m in mods:
        mod = importlib.import_module(m)
        out.append(f"{m} {getattr(mod, '__version__', '?')}")
    return ", ".join(out)


def check_python():
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10 or newer is needed, found {sys.version.split()[0]}")
    return sys.version.split()[0]


def demo_dir():
    env = os.environ.get("URBAN_DEMO_DIR")
    for d in ([Path(env)] if env else []) + [ROOT / "data" / "demo", ROOT / "data" / "outputs_v2" / "area"]:
        if (d / "tile_indicators.csv").exists() and (d / "footprints_all.geojson").exists():
            return d
    raise FileNotFoundError("no demo results found in data/demo (see the README)")


def check_demo_pipeline():
    import json

    import pandas as pd

    from src.analysis.insights import area_actions, brief_html, key_findings
    from src.analysis.scoring import score_tiles

    d = demo_dir()
    df = pd.read_csv(d / "tile_indicators.csv")
    feats = json.load(open(d / "footprints_all.geojson"))["features"]
    s = score_tiles(df)
    n_hi = int((s.tier == "High").sum())
    assert len(s) > 0 and s.score.notna().any(), "scoring produced no scores"
    assert len(key_findings(s)) >= 3 and len(area_actions(s)) >= 1
    html = brief_html(s, area_name="selfcheck")
    assert html.startswith("<!DOCTYPE") and "Where to act first" in html
    return f"{len(s)} blocks, {len(feats):,} building outlines, {n_hi} marked 'Act first', brief built"


def check_sample_tile():
    import json

    from src.analysis.pipeline import analyze_file, inspect_raster
    from src.model.predict import load_model

    tiles = sorted(SAMPLE_DIR.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError("no sample tiles in data/sample")
    tile = next((t for t in tiles if "img1028" in t.name), tiles[0])
    errors, _, info = inspect_raster(str(tile))
    assert not errors, errors
    model = load_model(str(WEIGHTS), "cpu")
    res = analyze_file(str(tile), model, json.load(open(STATS)), "cpu")
    row = res["df"].iloc[0]
    n, m2 = int(row.n_buildings), int(row.built_m2)
    msg = f"{tile.name}: {n} buildings, {m2:,} m² built-up"
    if "img1028" in tile.name:
        assert abs(n - EXPECT_N) <= 3, f"expected about {EXPECT_N} buildings, got {n}"
        assert abs(m2 - EXPECT_M2) <= 0.08 * EXPECT_M2, f"expected about {EXPECT_M2:,} m², got {m2:,}"
        msg += f" (reference: {EXPECT_N} and {EXPECT_M2:,} m²)"
    return msg


def main():
    print(f"Urban Footprints self-check · Python {sys.version.split()[0]} · {ROOT}\n")
    run("Python 3.10 or newer", check_python)
    run("Dashboard libraries", lambda: versions(["numpy", "pandas", "matplotlib", "PIL", "folium", "streamlit"]))
    run("Demo data, scoring, findings and planning brief", check_demo_pipeline)

    model_libs = ["torch", "segmentation_models_pytorch", "rasterio", "geopandas", "shapely", "scipy", "skimage"]
    missing = []
    for m in model_libs:
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001
            missing.append(m)
    if missing:
        skip("Model libraries", f"not installed or blocked: {', '.join(missing)} (needed for upload mode; "
                                f"see requirements-model.txt)")
        skip("Model on a sample tile", "model libraries missing")
    else:
        run("Model libraries", lambda: versions(model_libs))
        if not (WEIGHTS.exists() and STATS.exists()):
            skip("Model on a sample tile", "weights not found (download them from the GitHub Release into models/)")
        else:
            run("Model on a sample tile", check_sample_tile)

    print(f"\n{results['ok']} passed, {results['fail']} failed, {results['skip']} skipped")
    if results["fail"]:
        print("Something needs attention: see the [FAIL] lines above.")
    else:
        print("All checks that could run passed." + (" (Skipped items are optional.)" if results["skip"] else ""))
    sys.exit(1 if results["fail"] else 0)


if __name__ == "__main__":
    main()