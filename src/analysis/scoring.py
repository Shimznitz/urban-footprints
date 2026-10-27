# src/analysis/scoring.py
"""Settlement-pressure score and plain-language recommendations.

Everything here is rule-based and transparent. The score RANKS tiles against a reference area;
it is a screening aid built from image proxies, not a classification of informal settlements.
"""
import numpy as np
import pandas as pd

# indicator -> weight in the score, and which direction means "more pressure"
WEIGHTS = {"built_frac": 0.35, "density_per_ha": 0.25, "median_area_m2": 0.25, "veg_frac": 0.15}
DIRECTION = {"built_frac": +1, "density_per_ha": +1, "median_area_m2": -1, "veg_frac": -1}
MIN_BUILDINGS = 3          # fewer detected buildings than this: not enough to judge
MIN_VALID_M2 = 12_000      # less real imagery than this (about a third of a tile): not enough

DISCLAIMER = ("These are image-based proxies, not proof that an area is an informal settlement. "
              "Validate on the ground and with local data (tenure, services, community input) "
              "before acting.")


def _rank_vs(values, ref):
    """Share of the reference values that are LOWER than each value (ties split), 0..1."""
    ref = np.asarray(ref, float)
    ref = np.sort(ref[~np.isnan(ref)])
    values = np.asarray(values, float)
    if len(ref) == 0:
        return np.full(len(values), np.nan)
    lo = np.searchsorted(ref, values, side="left")
    hi = np.searchsorted(ref, values, side="right")
    out = (lo + hi) / 2 / len(ref)
    return np.where(np.isnan(values), np.nan, out)


def _raw_score(df, ref):
    ranks = pd.DataFrame(index=df.index)
    for k, sign in DIRECTION.items():
        r = _rank_vs(df[k].to_numpy(float), ref[k].to_numpy(float))
        ranks["r_" + k] = r if sign > 0 else 1 - r          # always: higher = more pressure
    w = np.array([WEIGHTS[k] for k in DIRECTION])
    R = ranks.to_numpy()
    num = np.nansum(R * w, axis=1)
    den = (~np.isnan(R) * w).sum(axis=1)
    score = np.where(den > 0, 100 * num / np.where(den > 0, den, 1), np.nan)
    return score, ranks


def score_tiles(df, ref=None, high_q=0.80, mod_q=0.50):
    """Add rank columns, a 0-100 score and a tier. Ranks are relative to `ref` (default: df itself)."""
    ref = df if ref is None else ref
    ref_scores, _ = _raw_score(ref, ref)
    t_high = np.nanquantile(ref_scores, high_q)               # top 20% of the reference = High
    t_mod = np.nanquantile(ref_scores, mod_q)                 # next 30% = Moderate
    score, ranks = _raw_score(df, ref)

    out = pd.concat([df.reset_index(drop=True), ranks.reset_index(drop=True)], axis=1)
    out["score"] = score
    tier = np.where(score >= t_high, "High", np.where(score >= t_mod, "Moderate", "Low"))
    thin = (out.n_buildings < MIN_BUILDINGS) | (out.valid_m2 < MIN_VALID_M2) | np.isnan(score)
    out["tier"] = np.where(thin, "Not enough data", tier)
    return out


def explain(row):
    """Plain-language reading of one scored tile. Returns dict(headline, drivers, advice, flags)."""
    headline = {
        "High": "Priority for field assessment (SDG 11.1: adequate housing and basic services).",
        "Moderate": "Watch list: some indicators are elevated.",
        "Low": "Lower priority compared with the rest of the analysed area.",
    }.get(row["tier"], "Too little imagery or too few detected buildings for a reliable reading.")

    drivers, advice = [], []
    if row["tier"] in ("High", "Moderate"):
        if row["r_built_frac"] >= 0.75:
            drivers.append(f"Built-up share is high ({row['built_frac']:.0%} of the area).")
            advice.append("Check road and fire-service access, drainage and open space on the ground.")
        if row["r_density_per_ha"] >= 0.75:
            drivers.append(f"Dense: about {row['density_per_ha']:.0f} buildings per hectare (counts are approximate).")
            advice.append("Check water, sanitation and electricity coverage for this density.")
        if row["r_median_area_m2"] >= 0.75:
            drivers.append(f"Typical building footprint is small ({row['median_area_m2']:.0f} m²).")
            advice.append("Check building condition and tenure security; small structures are a common but not conclusive sign.")
        if row["r_veg_frac"] >= 0.75:
            drivers.append(f"Little vegetation ({row['veg_frac']:.1%} of pixels).")
            advice.append("Consider shade trees and green space; low vegetation raises heat exposure.")
        if not drivers:
            drivers.append("No single indicator stands out; the score reflects a mix of moderately elevated values.")

    flags = []
    if row["mean_conf"] < 0.80:
        flags.append("Model confidence is lower than usual here; check the footprints before relying on them.")
    if row["valid_m2"] < MIN_VALID_M2:
        flags.append("Much of this tile has no imagery.")
    if row["n_buildings"] < MIN_BUILDINGS:
        flags.append("Very few buildings were detected, so the indicators are unreliable.")
    return dict(headline=headline, drivers=drivers, advice=advice, flags=flags)