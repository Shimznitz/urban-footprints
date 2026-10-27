# src/analysis/insights.py
"""Plain-language layer for policy makers: labels, key findings, area-level actions, and a
printable one-page briefing. Pure functions on the scored tile table (no Streamlit here)."""
import base64
import datetime
import html

import numpy as np
import pandas as pd

from src.analysis.scoring import DISCLAIMER

TIER_LABEL = {"High": "Act first", "Moderate": "Monitor", "Low": "Lower priority",
              "Not enough data": "Not enough data"}
TIER_COLOR = {"High": "#c0392b", "Moderate": "#d98a00", "Low": "#2e8b57", "Not enough data": "#7f8c8d"}
SIDE_NOTE = ("Rankings are relative: they show where conditions are most pressing compared with the "
             "rest of this area, not against an absolute standard.")
ACCURACY_LINE = ("Tested on Khartoum tiles the model never saw: building outlines matched real buildings "
                 "with an F1 score of about 0.60, and built-up area per block was typically within about "
                 "10-15% of the real area. Building counts are less reliable.")


def block_side_m(s):
    side = float(np.sqrt(s.valid_m2.max()))
    return int(round(side, -1)) if side >= 50 else int(round(side))


def centre(row):
    return (row["south"] + row["north"]) / 2, (row["west"] + row["east"]) / 2


def _word(rank, words):
    if pd.isna(rank):
        return None
    return words[0] if rank >= 0.8 else words[1] if rank >= 0.6 else words[2] if rank >= 0.4 else words[3]


def plain_lines(row):
    """Plain-language facts about one block, with the number in brackets."""
    lines = []
    w = _word(row["r_built_frac"], ["far more than elsewhere in this area", "more than most blocks",
                                    "about typical", "less than most blocks"])
    if w:
        lines.append(f"Buildings cover {row['built_frac']:.0%} of the land ({w}).")
    w = _word(row["r_density_per_ha"], ["far more crowded than elsewhere", "more crowded than most blocks",
                                        "typical crowding", "less crowded than most blocks"])
    if w:
        lines.append(f"About {row['density_per_ha']:.0f} buildings per hectare ({w}; counts are approximate).")
    w = _word(row["r_median_area_m2"], ["much smaller than elsewhere", "smaller than most",
                                        "typical size", "larger than most"])
    if w:
        lines.append(f"The typical building footprint is {row['median_area_m2']:.0f} m² ({w}).")
    w = _word(row["r_veg_frac"], ["far less green than elsewhere", "less green than most blocks",
                                  "typical greenery", "greener than most blocks"])
    if w:
        lines.append(f"Green cover is about {row['veg_frac']:.1%} of the land ({w}).")
    return lines


def key_findings(s):
    """A few plain sentences summarising the whole analysed area."""
    ok = s[s.tier != "Not enough data"]
    hi = ok[ok.tier == "High"]
    side = block_side_m(s)
    out = [f"We mapped {int(s.n_buildings.sum()):,} buildings covering {s.built_m2.sum() / 1e4:.1f} "
           f"hectares across {len(s)} blocks (each about {side} m × {side} m)."]
    if len(ok):
        out.append(f"{len(hi)} of {len(ok)} blocks ({len(hi) / len(ok):.0%}) are marked 'Act first'. "
                   f"They tend to combine heavy building cover, small buildings and little greenery, "
                   f"so field surveys should start there.")
    if len(hi):
        names = {"built_frac": "very high building cover", "density_per_ha": "crowding",
                 "median_area_m2": "small buildings", "veg_frac": "little greenery"}
        counts = {k: int((hi["r_" + k] >= 0.75).sum()) for k in names}
        top = [(k, c) for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:2] if c > 0]
        if top:
            out.append("In those blocks the most common warning signs are "
                       + " and ".join(f"{names[k]} ({c} blocks)" for k, c in top) + ".")
    if s.veg_frac.notna().any():
        out.append(f"Typical green cover is about {s.veg_frac.median():.1%} of the land "
                   f"(approximate, estimated from near-infrared light).")
    if len(ok):
        first = ok.sort_values("score", ascending=False).iloc[0]
        lat, lon = centre(first)
        out.append(f"The highest-ranked block is {first['tile']}, at about {lat:.4f}, {lon:.4f}.")
    thin = int((s.tier == "Not enough data").sum())
    if thin:
        out.append(f"{thin} block(s) had too little imagery or too few detected buildings to judge.")
    out.append(SIDE_NOTE)
    return out


def area_actions(s):
    """Suggested actions for the whole area, each with the blocks it applies to."""
    ok = s[s.tier != "Not enough data"]
    hi = ok[ok.tier == "High"]
    watch = ok[ok.tier.isin(["High", "Moderate"])]

    def ids(d):
        return d.sort_values("score", ascending=False)["tile"].tolist()

    acts = []
    if len(hi):
        acts.append(dict(
            title="Survey the 'Act first' blocks on the ground", sdg="SDG 11.1",
            body="Check building condition, tenure security and access to water, sanitation, electricity "
                 "and roads before deciding on upgrading or services.", blocks=ids(hi)))
    crowd = watch[(watch["r_built_frac"] >= 0.75) | (watch["r_density_per_ha"] >= 0.75)]
    if len(crowd):
        acts.append(dict(
            title="Plan access, drainage and fire safety in crowded blocks", sdg="SDG 11.1 · 11.5",
            body="Dense building cover can block emergency vehicles and water flow. Map lanes, drainage "
                 "paths and open spaces, and look for low-cost improvements.", blocks=ids(crowd)))
    small = watch[watch["r_median_area_m2"] >= 0.75]
    if len(small):
        acts.append(dict(
            title="Review housing quality and tenure where buildings are small", sdg="SDG 11.1",
            body="Small structures are a common but not conclusive sign of informal housing. Pair this map "
                 "with local records and community knowledge.", blocks=ids(small)))
    green = watch[watch["r_veg_frac"] >= 0.75]
    if len(green):
        acts.append(dict(
            title="Add shade and green space where cover is lowest", sdg="SDG 11.7",
            body="Low vegetation raises heat exposure. Look for vacant plots, road verges and school "
                 "grounds for trees and small parks.", blocks=ids(green)))
    acts.append(dict(
        title="Involve residents and verify before acting", sdg="SDG 11.3",
        body="Share these results with local government and community groups, and ground-check the "
             "flagged blocks. The map is a screening aid, not proof.", blocks=[]))
    return acts


def brief_html(s, images=None, area_name="Study area", title="Urban pressure briefing"):
    """A self-contained, printable HTML briefing. images: {tile: png bytes} for the top blocks."""
    images = images or {}
    e = html.escape
    ok = s[s.tier != "Not enough data"].sort_values("score", ascending=False)
    top = ok.head(5)

    findings = "".join(f"<li>{e(t)}</li>" for t in key_findings(s))
    actions = ""
    for a in area_actions(s):
        blocks = f"<div class='blocks'>Blocks: {e(', '.join(a['blocks'][:12]))}" \
                 f"{' …' if len(a['blocks']) > 12 else ''}</div>" if a["blocks"] else ""
        actions += (f"<div class='act'><b>{e(a['title'])}</b> <span class='sdg'>{e(a['sdg'])}</span>"
                    f"<p>{e(a['body'])}</p>{blocks}</div>")

    rows = ""
    for _, r in top.iterrows():
        lat, lon = centre(r)
        why = "<br>".join(e(x) for x in plain_lines(r)[:3])
        rows += (f"<tr><td><b>{e(str(r['tile']))}</b></td>"
                 f"<td><span class='chip' style='background:{TIER_COLOR[r['tier']]}'>{TIER_LABEL[r['tier']]}</span></td>"
                 f"<td>{lat:.4f}, {lon:.4f}</td><td>{why}</td></tr>")

    imgs = ""
    for tile in top.head(3)["tile"]:
        if tile in images:
            b64 = base64.b64encode(images[tile]).decode()
            imgs += (f"<figure><img src='data:image/png;base64,{b64}' alt='Block {e(str(tile))}'>"
                     f"<figcaption>Block {e(str(tile))}: red lines are detected buildings</figcaption></figure>")

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2b36;line-height:1.5}}
h1{{margin-bottom:.1rem}} h2{{margin-top:1.8rem;border-bottom:2px solid #145a8a;padding-bottom:.2rem;color:#145a8a}}
.sub{{color:#5b6b78;margin-top:0}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #d5dde3;padding:.5rem;text-align:left;vertical-align:top;font-size:.92rem}}
.chip{{color:#fff;padding:.1rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}}
.act{{border-left:4px solid #2a9d8f;background:#f1f8f7;padding:.4rem .9rem;margin:.6rem 0;border-radius:0 8px 8px 0}}
.act p{{margin:.2rem 0}} .sdg{{background:#145a8a;color:#fff;border-radius:6px;padding:0 .4rem;font-size:.75rem}}
.blocks{{font-size:.85rem;color:#5b6b78}} figure{{display:inline-block;margin:.4rem;width:30%;vertical-align:top}}
figure img{{width:100%;border-radius:8px}} figcaption{{font-size:.78rem;color:#5b6b78}}
.note{{background:#fff7e6;border:1px solid #f0d9a8;padding:.6rem .9rem;border-radius:8px;font-size:.9rem}}
@media print{{body{{margin:0}} h2{{break-after:avoid}} .act,figure,tr{{break-inside:avoid}}}}
</style></head><body>
<h1>{e(title)}</h1>
<p class="sub">{e(area_name)} · generated {datetime.date.today().isoformat()} · AI4SDGs 2026, Track 1 (SDG 11)</p>
<h2>What we found</h2><ul>{findings}</ul>
<h2>Where to act first</h2>
<table><tr><th>Block</th><th>Priority</th><th>Location (lat, lon)</th><th>Why</th></tr>{rows}</table>
<div>{imgs}</div>
<h2>Suggested actions</h2>{actions}
<h2>How far to trust this</h2>
<p>{e(ACCURACY_LINE)}</p>
<div class="note">{e(DISCLAIMER)}</div>
<p class="sub">Method: a U-Net neural network with a ResNet-50 encoder reads all 8 spectral bands and outlines
buildings; indicators are computed per block and ranked with a transparent score. Training data: SpaceNet 2,
Khartoum (CC BY-SA 4.0).</p>
</body></html>""" 