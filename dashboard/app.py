# dashboard/app.py
"""Urban Footprints dashboard (AI4SDGs 2026, Track 1).  Run from the project root:

    streamlit run dashboard/app.py

Written for decision makers: findings and recommended actions first, technical detail underneath.
Two modes: a precomputed Khartoum demo, and "analyse your own image" (drag and drop GeoTIFFs).
"""
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import branca.colormap as cm  # noqa: E402
import folium  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.insights import (  # noqa: E402
    ACCURACY_LINE, SIDE_NOTE, TIER_COLOR, TIER_LABEL, area_actions, brief_html, centre, key_findings,
    plain_lines)
from src.analysis.scoring import DISCLAIMER, WEIGHTS, explain, score_tiles  # noqa: E402

st.set_page_config(page_title="Urban Footprints | AI4SDGs Track 1", page_icon="🏙️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1250px}
.hero{padding:1.5rem 1.8rem;border-radius:18px;margin-bottom:1.1rem;color:#fff;
      background:linear-gradient(120deg,#0b3d2e 0%,#145a8a 100%)}
.hero h1{margin:0 0 .3rem 0;font-size:2.1rem;color:#fff;padding:0}
.hero p{margin:0;opacity:.93;font-size:1.05rem}
.kpi{border-radius:14px;padding:.8rem 1rem;background:rgba(128,128,128,.12);border:1px solid rgba(128,128,128,.25)}
.kpi .v{font-size:1.9rem;font-weight:700;line-height:1.15}.kpi .l{font-size:.85rem;opacity:.8}
.chip{display:inline-block;padding:.12rem .65rem;border-radius:999px;color:#fff;font-size:.8rem;font-weight:600}
.sdg{display:inline-block;padding:.05rem .5rem;border-radius:6px;background:#145a8a;color:#fff;font-size:.75rem;font-weight:600}
.finding{border-left:4px solid #2a9d8f;padding:.55rem .95rem;margin:.45rem 0;background:rgba(42,157,143,.12);
         border-radius:0 12px 12px 0}
.finding.note{border-left-color:#d98a00;background:rgba(217,138,0,.10);font-size:.92rem}
.sec{font-size:1.35rem;font-weight:700;margin:1.6rem 0 .3rem 0}
.sub{opacity:.75;margin:-.2rem 0 .6rem 0}
.small{font-size:.85rem;opacity:.8}
table.tbl{border-collapse:collapse;font-size:.9rem;width:100%}
table.tbl th,table.tbl td{border-bottom:1px solid #8884;padding:4px 8px;text-align:right}
table.tbl th:first-child,table.tbl td:first-child{text-align:left}
</style>""", unsafe_allow_html=True)

WEIGHTS_PATH = ROOT / "models" / "unet_resnet50_8band.pth"
STATS_PATH = ROOT / "models" / "band_stats.json"
SAMPLE_DIR = ROOT / "data" / "sample"
MAX_TOTAL_BLOCKS = 24

BAD_HIGH = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
GOOD_HIGH = ["#f7fcb9", "#addd8e", "#41ab5d", "#006837"]
NEUTRAL = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
MAP_VIEWS = {   # key -> (label, colour ramp or None for the categorical priority view)
    "tier": ("Priority level", None),
    "built_frac": ("Share of land covered by buildings", BAD_HIGH),
    "density_per_ha": ("Crowding (buildings per hectare)", BAD_HIGH),
    "median_area_m2": ("Typical building size (m²)", NEUTRAL),
    "veg_frac": ("Green cover", GOOD_HIGH),
    "score": ("Priority score (0-100)", BAD_HIGH),
}


# ----------------------------------------------------------------------------- helpers
def fmt(v, spec):
    return "-" if pd.isna(v) else format(v, spec)


def chip(tier):
    return f"<span class='chip' style='background:{TIER_COLOR[tier]}'>{TIER_LABEL[tier]}</span>"


def kpi_row(items):
    for col, (value, label) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div></div>",
                     unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f"<div class='sec'>{title}</div>" + (f"<div class='sub'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


def find_demo_dir():
    env = os.environ.get("URBAN_DEMO_DIR")
    for d in ([Path(env)] if env else []) + [ROOT / "data" / "demo", ROOT / "data" / "outputs_v2" / "area"]:
        if (d / "tile_indicators.csv").exists() and (d / "footprints_all.geojson").exists():
            return d
    return None


@st.cache_data(show_spinner="Loading results...")
def load_demo(d):
    d = Path(d)
    df = pd.read_csv(d / "tile_indicators.csv")
    feats = json.load(open(d / "footprints_all.geojson"))["features"]
    return df, feats


def group_by_tile(feats):
    out = {}
    for f in feats:
        out.setdefault(f["properties"]["tile"], []).append(f)
    return out


def html_table(view):
    # plain HTML on purpose: st.dataframe needs the compiled pyarrow package, which some
    # locked-down Windows machines block
    st.markdown(view.to_html(index=False, classes="tbl", border=0), unsafe_allow_html=True)


def parse_latlon(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) != 2:
        return None
    lat, lon = float(nums[0]), float(nums[1])
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def find_block(scored, lat, lon):
    hit = scored[(scored.west <= lon) & (lon <= scored.east) & (scored.south <= lat) & (lat <= scored.north)]
    return hit.iloc[0] if len(hit) else None


def embed_html(html, height):
    """st.iframe on current Streamlit; components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def make_map(scored, feats, view, show_buildings=False, overlays=None, center=None):
    south, north = scored.south.min(), scored.north.max()
    west, east = scored.west.min(), scored.east.max()
    m = folium.Map(location=list(center) if center else [(south + north) / 2, (west + east) / 2],
                   zoom_start=18 if center else 17, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite basemap (Esri)").add_to(m)
    for tile, img in (overlays or {}).items():
        r = scored.loc[scored.tile == tile].iloc[0]
        folium.raster_layers.ImageOverlay(image=img, bounds=[[r.south, r.west], [r.north, r.east]],
                                          opacity=0.95, name=f"Image {tile}", show=True).add_to(m)

    label, colors = MAP_VIEWS[view]
    cmap = None
    if colors:
        vals = scored[view].astype(float)
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        cmap = cm.LinearColormap(colors, vmin=lo, vmax=hi if hi > lo else lo + 1e-9, caption=label)

    cells = []
    for r in scored.itertuples():
        v = None if view == "tier" else getattr(r, view)
        cells.append({
            "type": "Feature",
            "properties": {
                "_v": None if v is None or pd.isna(v) else float(v), "_c": TIER_COLOR[r.tier],
                "Block": r.tile, "Priority": TIER_LABEL[r.tier],
                "Land covered by buildings": fmt(r.built_frac, ".0%"),
                "Buildings per hectare": fmt(r.density_per_ha, ".1f"),
                "Typical building size": fmt(r.median_area_m2, ".0f") + " m²",
                "Green cover": fmt(r.veg_frac, ".1%"),
            },
            "geometry": {"type": "Polygon", "coordinates": [[
                [r.west, r.south], [r.east, r.south], [r.east, r.north], [r.west, r.north], [r.west, r.south]]]},
        })

    def color(f):
        p = f["properties"]
        if view == "tier":
            return p["_c"]
        return "#999999" if p["_v"] is None else cmap(p["_v"])

    folium.GeoJson(
        {"type": "FeatureCollection", "features": cells}, name=f"Blocks: {label}",
        style_function=lambda f: {"fillColor": color(f), "color": "#333333", "weight": 0.6, "fillOpacity": 0.55},
        tooltip=folium.GeoJsonTooltip(fields=["Block", "Priority", "Land covered by buildings",
                                              "Buildings per hectare", "Typical building size", "Green cover"]),
    ).add_to(m)
    if cmap is not None:
        cmap.add_to(m)
    else:
        items = "".join(f"<div><span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
                        f"background:{TIER_COLOR[t]};margin-right:6px'></span>{TIER_LABEL[t]}</div>"
                        for t in ("High", "Moderate", "Low", "Not enough data"))
        m.get_root().html.add_child(folium.Element(
            "<div style='position:fixed;bottom:26px;left:12px;z-index:9999;background:rgba(255,255,255,.93);"
            "padding:8px 12px;border-radius:8px;font:13px sans-serif;color:#222'><b>Priority level</b>"
            + items + "</div>"))
    if center:
        folium.Marker(list(center), tooltip="Searched location").add_to(m)
    if show_buildings and feats:
        folium.GeoJson({"type": "FeatureCollection", "features": feats}, name="Building footprints",
                       style_function=lambda f: {"color": "#00e5ff", "weight": 1, "fillOpacity": 0.15}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    if not center:
        m.fit_bounds([[south, west], [north, east]])
    embed_html(m.get_root().render(), 600)


def tile_figure(row, feats, img, size=5.5):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(np.asarray(img), extent=(row.west, row.east, row.south, row.north))
    for f in feats:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for ring in poly:
                ax.plot([p[0] for p in ring], [p[1] for p in ring], color="#ff3b30", lw=0.8)
    ax.set_xlim(row.west, row.east)
    ax.set_ylim(row.south, row.north)
    ax.set_aspect(1 / np.cos(np.radians((row.south + row.north) / 2)))
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def tile_png(row, feats, img):
    fig = tile_figure(row, feats, img, size=4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


def tiles_table(scored):
    t = scored.sort_values("score", ascending=False)
    html_table(pd.DataFrame({
        "Block": t.tile, "Priority": t.tier.map(TIER_LABEL), "Score": t.score.map(lambda v: fmt(v, ".0f")),
        "Built-up": t.built_frac.map(lambda v: fmt(v, ".0%")),
        "Buildings/ha": t.density_per_ha.map(lambda v: fmt(v, ".1f")),
        "Median size (m²)": t.median_area_m2.map(lambda v: fmt(v, ".0f")),
        "Vegetation": t.veg_frac.map(lambda v: fmt(v, ".1%")),
    }))


def block_detail(scored, by_tile, tile, img):
    row = scored.loc[scored.tile == tile].iloc[0]
    ex = explain(row)
    left, right = st.columns(2)
    with left:
        fig = tile_figure(row, by_tile.get(tile, []), img)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red lines are the buildings the model detected.")
    with right:
        st.markdown(f"**Block {tile}** &nbsp; {chip(row.tier)}", unsafe_allow_html=True)
        st.write(ex["headline"])
        st.markdown("**What the image shows**\n" + "\n".join(f"- {l}" for l in plain_lines(row)))
        if ex["advice"]:
            st.markdown("**Suggested next steps**\n" + "\n".join(f"- {a}" for a in ex["advice"]))
        for fl in ex["flags"]:
            st.warning(fl)
        if "truth_m2" in row.index and not pd.isna(row.get("truth_m2")):
            st.caption(f"Check against real data: this validation block has {int(row.truth_n)} real buildings "
                       f"covering {int(row.truth_m2):,} m²; the model found {int(row.n_buildings)} covering "
                       f"{int(row.built_m2):,} m².")


def validation_section(df):
    if "truth_m2" not in df.columns or df.truth_m2.isna().all():
        return
    d = df.dropna(subset=["truth_m2"])
    d = d[d.truth_m2 > 0]
    r = float(np.corrcoef(d.truth_m2, d.built_m2)[0, 1])
    med_err = float(np.median(np.abs(d.built_m2 - d.truth_m2) / d.truth_m2))
    st.markdown(f"On these **{len(d)} validation blocks**, predicted built-up area correlates with the real "
                f"area at r = {r:.2f}, with a median error of {med_err:.0%} per block.")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, (x, y, title) in zip(ax, [
            (d.truth_m2 / 1e4, d.built_m2 / 1e4, "Built-up area per block (ha)"),
            (d.truth_n, d.n_buildings, "Building count per block")]):
        a.scatter(x, y, s=14, alpha=0.7)
        top = max(float(x.max()), float(y.max())) * 1.05
        a.plot([0, top], [0, top], "k--", lw=0.8)
        a.set_xlabel("real"); a.set_ylabel("predicted"); a.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


ACCURACY_TEXT = """
**Held-out test results (Khartoum, tiles the model never saw)**
- Pixel-level overlap (IoU): **0.69**.
- Object-level (a footprint counts only if it overlaps a real building by IoU ≥ 0.5): **F1 ≈ 0.60**
  (precision 0.61, recall 0.59) on 8 test tiles; the rough 95% interval is 0.55 to 0.66.
- Built-up **area** per tile was within about -9% to +16% of the real area; building **counts** ranged from
  -14% to +34%. That is why the ranking leans on area and treats counts as approximate.

**Known limits**
- Roofs that look like bare ground (large flat roofs, dark roofs) are under-detected, and touching
  buildings can merge into one outline. Outlines are smooth blobs, not exact rectangles.
- Trained on one city and one sensor (WorldView-3 at about 0.3 m). Other cities or sensors need checking.
- Vegetation uses an uncalibrated NDVI threshold of 0.2.
- The ranking compares blocks with a reference area. It is a screening aid, not a classification.
"""


# ----------------------------------------------------------------------------- shared results view
def results_view(scored, feats, by_tile, thumb_for, area_name, key, overlays=None, uid="x"):
    ok = scored[scored.tier != "Not enough data"]
    n_hi = int((scored.tier == "High").sum())
    kpi_row([(f"{int(scored.n_buildings.sum()):,}", "Buildings mapped"),
             (f"{scored.built_m2.sum() / 1e4:.1f} ha", "Built-up area"),
             (f"{len(scored)}", "Blocks analysed"),
             (f"{n_hi}", "Blocks to survey first")])

    section("What we found", "Plain-language summary of the whole area")
    for i, line in enumerate(key_findings(scored)):
        st.markdown(f"<div class='finding{' note' if line == SIDE_NOTE else ''}'>{line}</div>",
                    unsafe_allow_html=True)

    if len(ok):
        section("Where to act first", "The three highest-ranked blocks")
        top = ok.sort_values("score", ascending=False).head(3)
        for col, (i, (_, r)) in zip(st.columns(len(top)), enumerate(top.iterrows(), 1)):
            with col, st.container(border=True):
                st.markdown(f"**#{i} · Block {r.tile}** &nbsp; {chip(r.tier)}", unsafe_allow_html=True)
                img = thumb_for(r.tile)
                if img is not None:
                    fig = tile_figure(r, by_tile.get(r.tile, []), img, size=3.4)
                    st.pyplot(fig)
                    plt.close(fig)
                for line in plain_lines(r)[:3]:
                    st.markdown(f"<div class='small'>• {line}</div>", unsafe_allow_html=True)

    section("Recommended actions", "Suggested next steps for decision makers, with the blocks each applies to")
    for a in area_actions(scored):
        with st.container(border=True):
            st.markdown(f"**{a['title']}** &nbsp; <span class='sdg'>{a['sdg']}</span>", unsafe_allow_html=True)
            st.write(a["body"])
            if a["blocks"]:
                more = " …" if len(a["blocks"]) > 10 else ""
                st.markdown(f"<div class='small'>Blocks: {', '.join(a['blocks'][:10])}{more}</div>",
                            unsafe_allow_html=True)

    section("Map", "Hover over a block for its numbers. Use the layers button for satellite imagery.")
    c1, c2, c3 = st.columns([2, 2, 1])
    view = c1.selectbox("Colour the map by", list(MAP_VIEWS), format_func=lambda k: MAP_VIEWS[k][0],
                        key=f"{key}_view")
    loc = c2.text_input("Go to a place (latitude, longitude)", key=f"{key}_loc", placeholder="e.g. 15.55, 32.51")
    show_b = c3.checkbox("Building outlines", value=False, key=f"{key}_outl")
    center = None
    if loc:
        ll = parse_latlon(loc)
        if ll is None:
            st.warning("Please enter two numbers, latitude then longitude, as shown in Google Maps.")
        else:
            center = ll
            hit = find_block(scored, *ll)
            if hit is None:
                st.info("No analysed imagery covers this place yet. To analyse it, drop imagery of the area "
                        "into 'Analyse your own image' (or use the datasets the organisers provide).")
            else:
                st.success(f"This place is inside block {hit.tile} ({TIER_LABEL[hit.tier]}). "
                           f"Its details are shown below the map.")
                if st.session_state.get(f"{key}_lastloc") != loc:
                    st.session_state[f"{key}_tile_{uid}"] = hit.tile
                st.session_state[f"{key}_lastloc"] = loc
    make_map(scored, feats if show_b else [], view, show_buildings=show_b, overlays=overlays, center=center)

    section("Look at one block in detail")
    order = scored.sort_values("score", ascending=False)
    tile = st.selectbox("Block", order.tile.tolist(), key=f"{key}_tile_{uid}",
                        format_func=lambda t: f"{t} · {TIER_LABEL[order.loc[order.tile == t, 'tier'].iloc[0]]}")
    img = thumb_for(tile)
    if img is not None:
        block_detail(scored, by_tile, tile, img)
    else:
        st.info("No image stored for this block.")

    section("Take it with you")
    b1, b2 = st.columns([1, 2])
    if b1.button("Create planning brief", key=f"{key}_mkbrief"):
        images = {}
        for _, r in ok.sort_values("score", ascending=False).head(3).iterrows():
            im = thumb_for(r.tile)
            if im is not None:
                images[r.tile] = tile_png(r, by_tile.get(r.tile, []), im)
        st.session_state[f"{key}_brief"] = brief_html(scored, images, area_name=area_name)
    if st.session_state.get(f"{key}_brief"):
        b2.download_button("Download planning brief (open it in a browser, print to PDF)",
                           data=st.session_state[f"{key}_brief"], file_name="planning_brief.html",
                           mime="text/html", key=f"{key}_dlbrief")

    with st.expander("For analysts: indicator table and validation"):
        tiles_table(scored)
        validation_section(scored)
    with st.expander("How the ranking works and how accurate it is"):
        st.markdown(
            "Each block (about 190 m across) is ranked against a reference area on four indicators: "
            + ", ".join(f"{k.replace('_', ' ')} ({int(w * 100)}%)" for k, w in WEIGHTS.items())
            + ". More building cover, more crowding, smaller buildings and less vegetation all raise the "
              "ranking. The top 20% of blocks are marked **Act first**, the next 30% **Monitor**, the rest "
              "**Lower priority**.")
        st.markdown(ACCURACY_TEXT)
        st.caption(DISCLAIMER)


# ----------------------------------------------------------------------------- demo page
def page_demo():
    d = find_demo_dir()
    if d is None:
        st.error("No demo results found. Run `python -m src.analysis.run_area` first (see the README).")
        return
    df, feats = load_demo(str(d))
    scored = score_tiles(df)

    def thumb_for(t):
        p = d / "thumbs" / f"{t}.jpg"
        return Image.open(p) if p.exists() else None

    results_view(scored, feats, group_by_tile(feats), thumb_for, "Khartoum demo area", "demo")


# ----------------------------------------------------------------------------- upload page
@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    from src.model.predict import load_model
    return load_model(str(WEIGHTS_PATH), device), json.load(open(STATS_PATH)), device


def page_upload():
    section("Analyse your own imagery")
    st.write("Drag and drop one or more **8-band multispectral GeoTIFFs** (WorldView-3 band order, pixel size "
             "about 0.3 to 0.6 m, georeferenced). Large images are cut into tile-sized blocks.")
    if not (WEIGHTS_PATH.exists() and STATS_PATH.exists()):
        st.warning("Upload mode needs the trained weights at `models/unet_resnet50_8band.pth` and "
                   "`models/band_stats.json`. See the README for where to download them. "
                   "The Khartoum demo works without them.")
        return

    samples = sorted(SAMPLE_DIR.glob("*.tif")) if SAMPLE_DIR.exists() else []
    choice = "Upload GeoTIFFs"
    if samples:
        choice = st.radio("Image source", ["Upload GeoTIFFs", "Use a bundled sample tile"], horizontal=True)

    files = []            # list of (display name, path on disk)
    if choice == "Upload GeoTIFFs":
        ups = st.file_uploader("Drop .tif files here", type=["tif", "tiff"], accept_multiple_files=True)
        store = st.session_state.setdefault("upload_paths", {})
        for up in ups or []:
            k = (up.name, up.size)
            if k not in store:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                tmp.write(up.getvalue())
                tmp.close()
                store[k] = tmp.name
            files.append((up.name, store[k]))
    else:
        sample = st.selectbox("Sample tile", samples, format_func=lambda p: p.name)
        files.append((sample.name, str(sample)))
    if not files:
        return

    from src.analysis.pipeline import analyze_file, inspect_raster
    valid, total_blocks = [], 0
    for name, path in files:
        errors, warns, info = inspect_raster(path)
        if errors:
            for e in errors:
                st.error(f"{name}: {e}")
            continue
        for w in warns:
            st.warning(f"{name}: {w}")
        st.caption(f"{name}: {info['width']} × {info['height']} px · pixel size {info.get('pixel_m', '?')} m · "
                   f"{info['blocks']} block(s)")
        valid.append((name, path))
        total_blocks += info["blocks"]
    if not valid:
        return
    if total_blocks > MAX_TOTAL_BLOCKS:
        st.error(f"Together these files need {total_blocks} blocks; this app handles up to {MAX_TOTAL_BLOCKS} "
                 f"at a time. Please upload fewer or smaller files.")
        return

    key = tuple(sorted((n, p) for n, p in valid))
    st.write(f"Ready: {len(valid)} file(s), {total_blocks} block(s).")
    if st.button("Run analysis", type="primary"):
        model, stats, device = get_model()
        bar = st.progress(0.0, text=f"Starting on {device.upper()}...")
        done = {"n": 0}
        parts = []
        for name, path in valid:
            stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem)[:16].strip("_")
            prefix = f"{stem}_" if len(valid) > 1 else ""

            def prog(k, n):
                done["n"] += 1
                bar.progress(min(done["n"] / total_blocks, 1.0), text=f"Analysed block {done['n']} of {total_blocks}")

            parts.append(analyze_file(path, model, stats, device, progress=prog, prefix=prefix))
        bar.empty()
        res = dict(df=pd.concat([p["df"] for p in parts], ignore_index=True),
                   features=[f for p in parts for f in p["features"]],
                   thumbs={k: v for p in parts for k, v in p["thumbs"].items()})
        st.session_state["upload_result"] = {"key": key, "res": res}

    cached = st.session_state.get("upload_result")
    if cached and cached["key"] == key:
        show_upload_result(cached["res"])


def show_upload_result(res):
    d = find_demo_dir()
    ref_df = load_demo(str(d))[0] if d else None
    n = len(res["df"])
    ref = None
    if ref_df is not None:
        # a city should usually be ranked against itself; a handful of blocks needs a reference
        opts = ["Only the blocks in my upload", "The Khartoum reference area"]
        basis = st.radio("Rank the blocks against", opts, index=0 if n >= 10 else 1, horizontal=True,
                         key="up_basis")
        ref = None if basis == opts[0] else ref_df
    if ref is None and n < 5:
        if ref_df is not None:
            st.info("Fewer than 5 blocks is too few to rank against each other, so the Khartoum reference "
                    "area is used instead.")
            ref = ref_df
        else:
            st.info("Fewer than 5 blocks is too few for a meaningful ranking; read the numbers, not the tiers.")
    scored = score_tiles(res["df"], ref=ref)
    st.caption("Blocks are ranked against the Khartoum reference area." if ref is not None
               else "Blocks are ranked against each other (only the blocks in this upload).")
    uid = str(abs(hash(tuple(scored.tile))))[:8]
    results_view(scored, res["features"], group_by_tile(res["features"]), lambda t: res["thumbs"].get(t),
                 "Uploaded imagery", "up", overlays=res["thumbs"], uid=uid)
    st.download_button("Download building footprints (GeoJSON)",
                       data=json.dumps({"type": "FeatureCollection", "features": res["features"]}),
                       file_name="footprints.geojson", mime="application/geo+json")
    st.download_button("Download indicators (CSV)", data=scored.to_csv(index=False),
                       file_name="indicators.csv", mime="text/csv")


# ----------------------------------------------------------------------------- main
st.markdown(
    "<div class='hero'><h1>Urban Footprints</h1>"
    "<p>Satellite imagery in, a planning-ready picture of crowding, building cover and green space out. "
    "Built for informal-settlement assessment and urban planning · AI4SDGs 2026, Track 1 (SDG 11)</p></div>",
    unsafe_allow_html=True)
mode = st.sidebar.radio("Mode", ["Khartoum demo (precomputed)", "Analyse your own imagery"])
st.sidebar.markdown(
    "**How it works**\n\n1. A neural network (U-Net, ResNet-50) reads all 8 spectral bands and outlines every building\n"
    "2. We measure building cover, crowding, building size and greenery for each block\n"
    "3. Blocks are ranked with a transparent score, and plain-language actions are suggested\n\n"
    "Training data: SpaceNet 2, Khartoum (CC BY-SA 4.0).")
st.sidebar.caption(ACCURACY_LINE)
page_demo() if mode.startswith("Khartoum") else page_upload()