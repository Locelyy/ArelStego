"""
sweeping.py -- Multi-cover impact-ratio sweep for the hybrid GA + Shamir pipeline.

What it does
  1. Runs the full pipeline (GA group-average + Shamir LSB overflow) over a grid of
     impact ratios for EVERY cover image in ./Cover (up to MAX_COVERS).
  2. Finds the knee ratio PER COVER and on the MEAN curve across covers.
  3. Computes the "balanced hand-off" crossover (GA bits == Shamir bits) per cover.
  4. Marks the chosen operating ratio (CHOSEN_RATIO, default 0.25) on the figures.
  5. FIXED-RATIO MODE: runs ONE ratio on all covers and prints paper-ready numbers
     (mean +/- SD of PSNR, SSIM and the realised GA share across covers).

Usage
  python sweeping.py                  # full sweep over all covers (resumes finished ones)
  python sweeping.py --ratio 0.25     # only ratio 0.25 on all covers -> paper numbers
  python sweeping.py --max-covers 5   # quick trial on the first 5 covers
  python sweeping.py --force          # ignore cached per-cover results
  python sweeping.py --analyze-only   # re-plot / re-analyse from the saved CSV
  python sweeping.py --ratio 0.25 --analyze-only   # re-print the 0.25 tables from the CSV

Knee method
  Normalise GA capacity (x) and GA-only PSNR (y) to [0, 1], draw the chord from the
  first to the last point, and pick the ratio whose point is farthest from that chord.
  This is the maximum-distance-to-chord knee finder, closely related to Kneedle
  (Satopaa et al., 2011) but not identical: Kneedle works on the difference curve
  with a sensitivity parameter S. If `kneed` is installed (pip install kneed), the
  real Kneedle result is also reported as a cross-check column.

Notes on the reported numbers
  * Full PSNR / SSIM are measured on the first Shamir share (stego_final_list[0]).
  * Each cover's value is the average over all payload sizes; "mean +/- SD" is then
    taken across covers (each cover has equal weight).
"""
import os
import sys
import re
import io
import hashlib
import argparse
import contextlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

# Ensure UTF-8 output encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Ensure workspace is in sys.path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

import GAPS
from GAPS import (
    load_image_safe,
    embed_group_average_grey,
    embed_shamir_n_stego,
    calculate_metrics,
    PayloadProcessor,
    NUM_STEGO_IMAGES,
    SHAMIR_THRESHOLD
)

# ==========================================
# CONFIGURATION
# ==========================================
COVER_PATH = os.path.join(BASE_PATH, "Cover")
PAYLOAD_PATH = os.path.join(BASE_PATH, "payload")
RESULTS_DIR = os.path.join(BASE_PATH, "sweep_results")          # per-cover CSVs (resume)

OUTPUT_CSV = os.path.join(BASE_PATH, "sweeping_ratio_data.csv")  # all covers, all rows
OUTPUT_SUMMARY = os.path.join(BASE_PATH, "sweeping_knee_summary.csv")
OUTPUT_GRAPH = os.path.join(BASE_PATH, "sweeping_ratio_tradeoff.png")
OUTPUT_KNEE_GRAPH = os.path.join(BASE_PATH, "sweeping_knee_distribution.png")

COVER_EXTENSIONS = (".tiff", ".tif", ".png", ".bmp", ".jpg", ".jpeg", ".pgm")
MAX_COVERS = 15            # None = use every image found in COVER_PATH

# Operating point used in the paper: the (approximately) balanced GA / Shamir hand-off,
# i.e. the mean-curve crossover (~0.256) rounded to 0.25. Only affects figure markers;
# exact numbers at this ratio come from `python sweeping.py --ratio 0.25`.
CHOSEN_RATIO = 0.25

# Sweep from 0.00037 (architectural 32-bit minimum) up to 1.0 (100% of candidates).
# NOTE: keep this list unchanged if you already have sweep_results/ cached -- changing it
# invalidates the cache and forces a full re-run. Use --ratio for any single extra ratio.
RATIOS = [
    0.00037, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.1, 0.14, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

# Standard evaluation payload sizes (in KB)
PAYLOAD_SIZES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

KNEE_Y_COLUMN = "GA Only PSNR (dB)"   # curve the knee is computed on (vs. GA capacity)
SHOW_CROSSOVER = True                  # draw the mean-curve crossover if it differs from CHOSEN_RATIO

METRIC_COLS = [
    "GA Bits", "Shamir Overflow Bits",
    "Full PSNR (dB)", "Full SSIM",
    "GA Only PSNR (dB)", "GA Only SSIM",
]

# ==========================================
# PERFORMANCE OPTIMIZATION: CACHE IMPACT MAP
# ==========================================
# NOTE: the key hashes the FULL channel. The previous key used only the first 256
# bytes, which can collide between different covers (e.g. similar borders) and would
# silently reuse the wrong impact map in a multi-cover run.
_impact_cache = {}
_orig_calc_impact = GAPS.calculate_pixel_impact


def _cached_calculate_pixel_impact(channel, window_size=3):
    digest = hashlib.blake2b(np.ascontiguousarray(channel).tobytes(), digest_size=16).hexdigest()
    key = (digest, channel.shape, str(channel.dtype), window_size)
    if key not in _impact_cache:
        _impact_cache[key] = _orig_calc_impact(channel, window_size)
    return _impact_cache[key]


GAPS.calculate_pixel_impact = _cached_calculate_pixel_impact


@contextlib.contextmanager
def quiet():
    """Silence stdout from the embedding routines."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ==========================================
# KNEE / CROSSOVER ANALYSIS
# ==========================================
def find_knee(curve, y_col=KNEE_Y_COLUMN, verbose=False, title=""):
    """
    Maximum-distance-to-chord knee on (GA capacity, y_col), both normalised to [0, 1].
    `curve` is a DataFrame with one row per ratio. Returns None if there are < 3 valid points.
    """
    d = (curve.dropna(subset=["GA Bits", y_col])
              .sort_values("Ratio").reset_index(drop=True))
    if len(d) < 3:
        return None

    ratios = d["Ratio"].values
    cap = d["GA Bits"].values.astype(float)
    y = d[y_col].values.astype(float)

    x_norm = (cap - cap.min()) / (cap.max() - cap.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    pts = np.column_stack([x_norm, y_norm])
    p_start, p_end = pts[0], pts[-1]
    secant = p_end - p_start
    secant = secant / (np.linalg.norm(secant) + 1e-10)

    diff = pts - p_start
    dist = np.abs(secant[0] * diff[:, 1] - secant[1] * diff[:, 0])
    best = int(np.argmax(dist))

    if verbose:
        full_p = d["Full PSNR (dB)"].values
        full_s = d["Full SSIM"].values
        print("\n" + "=" * 85)
        print(f"🎯 KNEE DISCOVERY {title}")
        print("=" * 85)
        print(f" {'Ratio':>9} | {'GA Capacity':>11} | {'GA PSNR':>9} | {'Full PSNR':>10} | {'Full SSIM':>10} | {'Knee Dist':>10}")
        print("-" * 85)
        for i in range(len(d)):
            mark = "  ★ KNEE" if i == best else ""
            print(f" {ratios[i]:9.5f} | {int(cap[i]):11,d} | {y[i]:9.2f} | {full_p[i]:10.2f} | {full_s[i]:10.5f} | {dist[i]:10.4f}{mark}")
        print("=" * 85)
        print(f"🏆 Knee ratio: {ratios[best]} ({ratios[best] * 100:.2f}%)  |  d_knee = {dist[best]:.4f}")
        print("=" * 85 + "\n")

    return {
        "best_idx": best,
        "ratio": float(ratios[best]),
        "distance": float(dist[best]),
        "distances": dist,
        "full_psnr": float(d["Full PSNR (dB)"].iloc[best]),
        "full_ssim": float(d["Full SSIM"].iloc[best]),
        "ga_bits": float(d["GA Bits"].iloc[best]),
        "shamir_bits": float(d["Shamir Overflow Bits"].iloc[best]),
    }


def kneed_cross_check(curve, S=1.0):
    """Optional: real Kneedle via the `kneed` package. Returns NaN if unavailable/undetected."""
    try:
        from kneed import KneeLocator
    except ImportError:
        return np.nan
    d = (curve.dropna(subset=["GA Bits", KNEE_Y_COLUMN])
              .sort_values("Ratio").reset_index(drop=True))
    if len(d) < 3:
        return np.nan
    try:
        kl = KneeLocator(d["GA Bits"].values, d[KNEE_Y_COLUMN].values,
                         S=S, curve="convex", direction="decreasing")
        if kl.knee is None:
            return np.nan
        idx = int(np.argmin(np.abs(d["GA Bits"].values - kl.knee)))
        return float(d["Ratio"].values[idx])
    except Exception:
        return np.nan


def find_crossover(curve):
    """Ratio where GA bits first reach Shamir bits (log-ratio interpolation). NaN if none."""
    d = (curve.dropna(subset=["GA Bits", "Shamir Overflow Bits"])
              .sort_values("Ratio").reset_index(drop=True))
    if len(d) < 2:
        return np.nan
    x = np.log10(d["Ratio"].values)
    diff = d["GA Bits"].values - d["Shamir Overflow Bits"].values
    for i in range(len(diff) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(10 ** (x[i] + t * (x[i + 1] - x[i])))
    return np.nan


def interp_at(curve, ratio, col):
    d = curve.dropna(subset=[col]).sort_values("Ratio")
    return float(np.interp(np.log10(ratio), np.log10(d["Ratio"].values), d[col].values))


# ==========================================
# SWEEP
# ==========================================
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def discover_covers(max_covers):
    if not os.path.isdir(COVER_PATH):
        return []
    files = [os.path.join(COVER_PATH, f) for f in os.listdir(COVER_PATH)
             if f.lower().endswith(COVER_EXTENSIONS)]
    files.sort(key=lambda p: natural_key(os.path.basename(p)))
    return files[:max_covers] if max_covers else files


def load_payloads():
    processor = PayloadProcessor()
    payloads = {}
    for sz in PAYLOAD_SIZES:
        p_path = os.path.join(PAYLOAD_PATH, f"random-binary_{sz}Kb.txt")
        if not os.path.exists(p_path):
            print(f"⚠️  Payload file missing, skipped: {p_path}")
            continue
        with open(p_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        with quiet():
            bits, _ = processor.process_for_embedding(txt)
        payloads[sz] = bits
    return payloads


def _load_cached_cover(csv_path, payloads, ratios):
    """Return cached per-cover results if they are complete and error-free, else None."""
    if not os.path.exists(csv_path):
        return None
    try:
        d = pd.read_csv(csv_path)
        if set(np.round(d["Ratio"].unique(), 8)) != set(np.round(ratios, 8)):
            return None
        if set(d["Payload (KB)"].unique()) != set(payloads.keys()):
            return None
        if d["Full PSNR (dB)"].isna().any():
            return None
        return d
    except Exception:
        return None


def sweep_one_cover(cover_file, payloads, ratios):
    name = os.path.basename(cover_file)
    cover_img = load_image_safe(cover_file)
    if cover_img is None:
        print(f"❌ Could not load cover {cover_file}; skipped.")
        return None

    flat = cover_img.flatten()
    cand_count = int(((flat >= 5) & (flat <= 249)).sum())
    print(f"\n📸 Cover: {name} ({cand_count:,} candidate pixels)")

    rows = []
    for r in ratios:
        print(f"   Ratio {r:<8} ({r * 100:6.3f}%) ...", end=" ", flush=True)
        r_psnr, r_ssim = [], []
        n_fail, first_err = 0, None

        for sz, bits in payloads.items():
            total_bits = len(bits)
            try:
                with quiet():
                    # Stage 1: Adaptive Group Average Embedding
                    stego_ga, indices_ga, _, bits_in_ga = embed_group_average_grey(
                        cover_img, bits, use_impact_analysis=True, impact_pool_ratio=r
                    )
                    m_ga = calculate_metrics(f"GA_r{r}", cover_img, stego_ga, bits_in_ga)

                    # Stage 2: Shamir Secret Sharing LSB for overflow
                    bits_remaining = total_bits - bits_in_ga
                    if bits_remaining > 0:
                        stego_final_list, _ = embed_shamir_n_stego(
                            stego_ga, bits[bits_in_ga:], NUM_STEGO_IMAGES, SHAMIR_THRESHOLD,
                            exclusion_indices=indices_ga
                        )
                    else:
                        stego_final_list = [stego_ga.copy() for _ in range(NUM_STEGO_IMAGES)]

                    m_full = calculate_metrics(f"Full_r{r}", cover_img, stego_final_list[0], total_bits)

                row = {
                    "GA Bits": bits_in_ga,
                    "Shamir Overflow Bits": max(0, bits_remaining),
                    "Full PSNR (dB)": m_full['psnr'],
                    "Full SSIM": m_full['ssim'],
                    "GA Only PSNR (dB)": m_ga['psnr'],
                    "GA Only SSIM": m_ga['ssim'],
                }
            except Exception as e:
                # Failures become NaN (NOT 0) so they cannot drag the averages down.
                n_fail += 1
                first_err = first_err or f"{type(e).__name__}: {e}"
                row = {c: np.nan for c in METRIC_COLS}

            r_psnr.append(row["Full PSNR (dB)"])
            r_ssim.append(row["Full SSIM"])
            rows.append({
                "Cover": name, "Ratio": r, "Ratio (%)": r * 100,
                "Payload (KB)": sz, "Total Bits": total_bits, **row,
            })

        mp, ms = pd.Series(r_psnr).mean(), pd.Series(r_ssim).mean()
        msg = f"Full PSNR = {mp:.2f} dB, Full SSIM = {ms:.6f}"
        if n_fail:
            msg += f"  ⚠️ {n_fail} failed ({first_err})"
        print(msg)

    _impact_cache.clear()   # bound memory between covers
    return pd.DataFrame(rows)


def run_sweep(max_covers, force=False, ratios=None, results_dir=None, output_csv=None):
    ratios = list(ratios) if ratios is not None else RATIOS
    results_dir = results_dir or RESULTS_DIR
    output_csv = output_csv or OUTPUT_CSV

    covers = discover_covers(max_covers)
    if not covers:
        print(f"❌ No cover images found in {COVER_PATH}")
        return None
    payloads = load_payloads()
    if not payloads:
        print(f"❌ No payload files found in {PAYLOAD_PATH}")
        return None

    os.makedirs(results_dir, exist_ok=True)
    print("=" * 85)
    print("🚀 Multi-Cover Ratio Sweep (GA + Shamir Secret Sharing)")
    print(f"   Covers: {len(covers)} | Ratios: {len(ratios)} | Payload sizes: {len(payloads)}")
    print(f"   Shamir configuration: (k={SHAMIR_THRESHOLD}, n={NUM_STEGO_IMAGES})")
    print("=" * 85)

    frames = []
    for i, path in enumerate(covers, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        csv_path = os.path.join(results_dir, f"{stem}.csv")
        print(f"\n[{i}/{len(covers)}]", end=" ")

        cached = None if force else _load_cached_cover(csv_path, payloads, ratios)
        if cached is not None:
            print(f"♻️  {os.path.basename(path)}: using cached results ({csv_path})")
            frames.append(cached)
            continue

        df_cover = sweep_one_cover(path, payloads, ratios)
        if df_cover is not None:
            df_cover.to_csv(csv_path, index=False)   # checkpoint after every cover
            frames.append(df_cover)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"\n💾 All data saved to: {output_csv}")
    return df


# ==========================================
# FIXED-RATIO MODE (paper numbers for one chosen ratio)
# ==========================================
def single_ratio_paths(ratio):
    tag = f"{ratio:g}"
    return {
        "dir": os.path.join(RESULTS_DIR, f"ratio_{tag}"),
        "data": os.path.join(BASE_PATH, f"sweeping_ratio_{tag}_data.csv"),
        "cover": os.path.join(BASE_PATH, f"sweeping_ratio_{tag}_per_cover.csv"),
        "payload": os.path.join(BASE_PATH, f"sweeping_ratio_{tag}_by_payload.csv"),
    }


def report_single_ratio(df, ratio):
    """Per-cover table + mean +/- SD across covers + realised GA share + per-payload table."""
    paths = single_ratio_paths(ratio)
    d = df.copy()
    d["GA Share (%)"] = 100.0 * d["GA Bits"] / d["Total Bits"]
    cols = ["Full PSNR (dB)", "Full SSIM", "GA Only PSNR (dB)", "GA Only SSIM",
            "GA Bits", "Shamir Overflow Bits", "GA Share (%)"]

    n_nan = int(d["Full PSNR (dB)"].isna().sum())
    if n_nan:
        print(f"⚠️  {n_nan} of {len(d)} runs failed and were excluded from the averages (NaN).")

    per_cover = d.groupby("Cover", as_index=False)[cols].mean()
    order = sorted(per_cover["Cover"], key=natural_key)
    per_cover = per_cover.set_index("Cover").loc[order].reset_index()
    n = len(per_cover)
    n_pay = d["Payload (KB)"].nunique()

    print("\n" + "=" * 85)
    print(f"📌 FIXED-RATIO RESULTS AT r = {ratio:g}   ({n} covers, each averaged over {n_pay} payload sizes)")
    print("=" * 85)
    show = per_cover[["Cover", "Full PSNR (dB)", "Full SSIM", "GA Share (%)", "GA Bits", "Shamir Overflow Bits"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\nMean ± SD across covers (range):")
    for label, col, fmt in [("PSNR (dB)", "Full PSNR (dB)", ".2f"),
                            ("SSIM", "Full SSIM", ".5f"),
                            ("GA share (%)", "GA Share (%)", ".1f")]:
        s = per_cover[col]
        sd = s.std(ddof=1) if n > 1 else 0.0
        print(f"   {label:<13}: {s.mean():{fmt}} ± {sd:{fmt}}   (range {s.min():{fmt}} – {s.max():{fmt}})")

    share = per_cover["GA Share (%)"]
    near = int(((share >= 40) & (share <= 60)).sum())
    print(f"   Covers with GA share in 40-60%: {near} of {n}")
    print("   (Full PSNR/SSIM are measured on the first Shamir share, as in the sweep.)")

    by_payload = d.groupby("Payload (KB)")[cols].mean().reset_index()
    print("\nBy payload size (mean over covers):")
    print(by_payload[["Payload (KB)", "Full PSNR (dB)", "Full SSIM", "GA Share (%)"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("=" * 85)

    per_cover.to_csv(paths["cover"], index=False)
    by_payload.to_csv(paths["payload"], index=False)
    print(f"💾 Per-cover table:   {paths['cover']}")
    print(f"💾 By-payload table:  {paths['payload']}")


def run_fixed_ratio(ratio, max_covers, force=False, analyze_only=False):
    paths = single_ratio_paths(ratio)
    if analyze_only:
        if not os.path.exists(paths["data"]):
            print(f"❌ {paths['data']} not found; run `python sweeping.py --ratio {ratio:g}` first.")
            return
        df = pd.read_csv(paths["data"])
    else:
        df = run_sweep(max_covers, force=force, ratios=[ratio],
                       results_dir=paths["dir"], output_csv=paths["data"])
    if df is None:
        return
    report_single_ratio(df, ratio)


# ==========================================
# AGGREGATION + SUMMARY
# ==========================================
def aggregate(df):
    """Average over payloads within each cover, then mean/SD across covers."""
    per_cover = df.groupby(["Cover", "Ratio"], as_index=False)[METRIC_COLS].mean()
    g = per_cover.groupby("Ratio")[METRIC_COLS]
    mean_curve = g.mean().reset_index()
    std_curve = g.std(ddof=1).fillna(0.0).reset_index()
    return per_cover, mean_curve, std_curve


def build_summary(per_cover):
    rows = []
    for cover, c in per_cover.groupby("Cover"):
        k = find_knee(c)
        k_full = find_knee(c, "Full PSNR (dB)")
        rows.append({
            "Cover": cover,
            "Knee Ratio": k["ratio"] if k else np.nan,
            "d_knee": k["distance"] if k else np.nan,
            "Full PSNR at Knee (dB)": k["full_psnr"] if k else np.nan,
            "Full SSIM at Knee": k["full_ssim"] if k else np.nan,
            "GA Bits at Knee": k["ga_bits"] if k else np.nan,
            "Knee Ratio (Full-PSNR curve)": k_full["ratio"] if k_full else np.nan,
            "Knee Ratio (kneed lib)": kneed_cross_check(c),
            "Crossover Ratio (GA = Shamir)": find_crossover(c),
        })
    return pd.DataFrame(rows)


def print_summary(summary, mean_knee, mean_cross, mean_curve):
    print("\n" + "=" * 85)
    print("📊 PER-COVER KNEE SUMMARY")
    print("=" * 85)
    show = summary[["Cover", "Knee Ratio", "d_knee", "Full PSNR at Knee (dB)",
                    "Full SSIM at Knee", "Crossover Ratio (GA = Shamir)"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    kr = summary["Knee Ratio"].dropna()
    cr = summary["Crossover Ratio (GA = Shamir)"].dropna()
    if len(kr):
        print("\nKnee ratio across covers:")
        print(f"   n = {len(kr)} | median = {kr.median():.4g} | mean = {kr.mean():.4g} | "
              f"geo-mean = {np.exp(np.log(kr).mean()):.4g} | min = {kr.min():.4g} | max = {kr.max():.4g}")
        print("   counts per ratio: " + ", ".join(f"{k:g}: {v}" for k, v in kr.value_counts().sort_index().items()))
        if mean_knee:
            frac = float((np.isclose(kr, mean_knee["ratio"])).mean())
            print(f"   fraction of covers whose knee equals the mean-curve knee ({mean_knee['ratio']:g}): {frac:.0%}")
    if len(cr):
        print(f"Crossover ratio across covers: median = {cr.median():.4g} | "
              f"min = {cr.min():.4g} | max = {cr.max():.4g}")
    if not np.isnan(mean_cross):
        print(f"Mean-curve crossover ≈ {mean_cross:.4f}  "
              f"(PSNR ≈ {interp_at(mean_curve, mean_cross, 'Full PSNR (dB)'):.2f} dB, "
              f"SSIM ≈ {interp_at(mean_curve, mean_cross, 'Full SSIM'):.5f})")
    if CHOSEN_RATIO is not None:
        print(f"Chosen ratio {CHOSEN_RATIO:g}: interpolated estimate "
              f"PSNR ≈ {interp_at(mean_curve, CHOSEN_RATIO, 'Full PSNR (dB)'):.2f} dB, "
              f"SSIM ≈ {interp_at(mean_curve, CHOSEN_RATIO, 'Full SSIM'):.5f}  "
              f"(exact values: python sweeping.py --ratio {CHOSEN_RATIO:g})")
    print("=" * 85)


# ==========================================
# FIGURES
# ==========================================
def generate_tradeoff_figure(mean_curve, std_curve, knee, crossover, n_covers,
                             chosen_ratio=None, knee_range=None):
    """
    Clean two-panel figure (publication style).
      (a) Full-pipeline PSNR (mean +/- 1 SD across covers)
      (b) Bits embedded in the GA safe pool vs. the Shamir LSB overflow
    Titles are intentionally omitted: put them in the LaTeX/Word caption.
    `crossover` and `knee_range` are kept in the signature for compatibility but unused.
    SSIM and GA-only curves are exported to a table instead of being plotted.
    """
    print("🎨 Rendering trade-off figure...")
    x = mean_curve["Ratio"]
    k_ratio = knee["ratio"]

    blue, orange = '#1f77b4', '#e65100'
    gray, purple = '#555555', '#6a1b9a'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0), facecolor='white', sharex=True)
    fig.subplots_adjust(hspace=0.12)

    # ---------- (a) quality ----------
    mp, sp = mean_curve["Full PSNR (dB)"], std_curve["Full PSNR (dB)"]
    ax1.plot(x, mp, marker='o', markersize=4, linewidth=1.8, color=blue,
             label='Full-pipeline PSNR')
    ax1.fill_between(x, mp - sp, mp + sp, color=blue, alpha=0.15, linewidth=0)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_title("(a)", loc='left', fontsize=11, fontweight='bold')

    # ---------- (b) payload allocation ----------
    ga, sh = mean_curve["GA Bits"], mean_curve["Shamir Overflow Bits"]
    ax2.plot(x, ga, marker='o', markersize=4, linewidth=1.8, color=blue,
             label='GA safe pool')
    ax2.plot(x, sh, marker='o', markersize=4, linewidth=1.8, linestyle='--', color=orange,
             label='Shamir LSB overflow')
    ax2.set_ylabel("Average bits embedded", fontsize=11)
    ax2.set_xlabel("Impact ratio (log scale)", fontsize=11)
    ax2.set_title("(b)", loc='left', fontsize=11, fontweight='bold')

    # ---------- reference lines (labelled once, in panel a) ----------
    for ax in (ax1, ax2):
        ax.axvline(k_ratio, color=gray, linestyle='--', linewidth=1.4,
                   label=f"Knee of GA-only PSNR ({k_ratio:g})" if ax is ax1 else None)
        if chosen_ratio is not None:
            ax.axvline(chosen_ratio, color=purple, linewidth=2.0,
                       label=f"Chosen ratio ({chosen_ratio:g})" if ax is ax1 else None)
        ax.grid(True, which='major', linestyle=':', alpha=0.5)

    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.legend(loc='center left', fontsize=9, framealpha=0.9)

    # ---------- x axis: few, plain numeric ticks ----------
    ax2.set_xscale('log')
    ticks = [0.00037, 0.001, 0.01, 0.07, 0.25, 0.5, 1.0]
    if chosen_ratio is not None:
        ticks.append(chosen_ratio)
    ticks = sorted(set(ticks))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9.5)
    ax2.xaxis.set_minor_formatter(NullFormatter())

    plt.savefig(OUTPUT_GRAPH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Trade-off figure saved to: {OUTPUT_GRAPH}")

    # ---------- operating-point table (replaces SSIM / GA-only curves) ----------
    pts = []
    for label, r in [("Knee", k_ratio), ("Chosen", chosen_ratio), ("All", 1.0)]:
        if r is None:
            continue
        pts.append({
            "Point": label,
            "Ratio": r,
            "GA Bits": interp_at(mean_curve, r, "GA Bits"),
            "Shamir Bits": interp_at(mean_curve, r, "Shamir Overflow Bits"),
            "Full PSNR (dB)": interp_at(mean_curve, r, "Full PSNR (dB)"),
            "Full SSIM": interp_at(mean_curve, r, "Full SSIM"),
            "GA-only PSNR (dB)": interp_at(mean_curve, r, "GA Only PSNR (dB)"),
            "GA-only SSIM": interp_at(mean_curve, r, "GA Only SSIM"),
        })
    pts_df = pd.DataFrame(pts)
    pts_path = os.path.join(BASE_PATH, "sweeping_operating_points.csv")
    pts_df.to_csv(pts_path, index=False)
    print("\nOperating points (mean over covers):")
    print(pts_df.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print(f"💾 Operating-point table saved to: {pts_path}")
    print(f"\nSuggested caption: Impact-ratio sweep (mean of {n_covers} covers). "
          f"(a) Full-pipeline PSNR, shaded band = ±1 SD. (b) Payload bits carried by the GA "
          f"safe pool and the Shamir LSB overflow. Dashed line: knee of GA-only PSNR vs. GA "
          f"capacity; solid line: chosen ratio.")

def generate_knee_distribution_figure(summary, mean_knee_ratio, mean_cross, chosen_ratio=None):
    s = summary.sort_values("Knee Ratio").reset_index(drop=True)
    n = len(s)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * n + 1.8)), facecolor='white')
    y = np.arange(n)

    ax.scatter(s["Knee Ratio"], y, s=90, color='#2ca02c', edgecolor='black', zorder=3,
               label='Knee ratio (per cover)')
    cr = s["Crossover Ratio (GA = Shamir)"]
    ax.scatter(cr, y, s=70, marker='D', color='#e65100', edgecolor='black', zorder=3,
               label='Balanced crossover (per cover)')

    kr_med = s["Knee Ratio"].median()
    ax.axvline(kr_med, color='#2ca02c', linestyle='--', linewidth=1.8,
               label=f"Median knee = {kr_med:.3g}")
    if cr.notna().any():
        ax.axvline(cr.median(), color='#e65100', linestyle='--', linewidth=1.8,
                   label=f"Median crossover = {cr.median():.3g}")
    if mean_knee_ratio is not None:
        ax.axvline(mean_knee_ratio, color='#2ca02c', linestyle=':', linewidth=1.8,
                   label=f"Mean-curve knee = {mean_knee_ratio:g}")
    if chosen_ratio is not None:
        ax.axvline(chosen_ratio, color='#6a1b9a', linewidth=2.4, alpha=0.9,
                   label=f"Chosen ratio = {chosen_ratio:g}")

    ax.set_xscale('log')
    ticks = sorted(set([0.05, 0.1, 0.2, 0.5] + ([chosen_ratio] if chosen_ratio is not None else [])))
    # keep the tick labels from colliding (0.2 vs 0.25)
    if chosen_ratio is not None and 0.2 in ticks and np.isclose(chosen_ratio, 0.25):
        ticks.remove(0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([os.path.splitext(c)[0] for c in s["Cover"]], fontsize=9)
    ax.set_xlabel("Impact Ratio (log scale)", fontsize=11, fontweight='semibold')
    ax.set_title(f"Per-Cover Knee vs. Balanced Crossover (n = {n})", fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUTPUT_KNEE_GRAPH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Knee-distribution figure saved to: {OUTPUT_KNEE_GRAPH}")


# ==========================================
# MAIN
# ==========================================
def analyze(df):
    per_cover, mean_curve, std_curve = aggregate(df)
    n_covers = per_cover["Cover"].nunique()

    n_nan = int(df["Full PSNR (dB)"].isna().sum())
    if n_nan:
        print(f"⚠️  {n_nan} of {len(df)} runs failed and were excluded from the averages (NaN).")

    mean_knee = find_knee(mean_curve, verbose=True, title=f"(mean curve across {n_covers} covers)")
    if mean_knee is None:
        print("❌ Not enough valid data points to locate a knee.")
        return

    mean_cross = find_crossover(mean_curve)
    summary = build_summary(per_cover)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"💾 Per-cover summary saved to: {OUTPUT_SUMMARY}")
    print_summary(summary, mean_knee, mean_cross, mean_curve)

    kr = summary["Knee Ratio"].dropna()
    knee_range = (float(kr.min()), float(kr.max())) if len(kr) > 1 else None
    generate_tradeoff_figure(mean_curve, std_curve, mean_knee, mean_cross, n_covers,
                             chosen_ratio=CHOSEN_RATIO, knee_range=knee_range)
    if n_covers > 1:
        generate_knee_distribution_figure(summary, mean_knee["ratio"], mean_cross,
                                          chosen_ratio=CHOSEN_RATIO)


def main():
    ap = argparse.ArgumentParser(description="Multi-cover impact-ratio sweep (GA + Shamir).")
    ap.add_argument("--max-covers", type=int, default=MAX_COVERS,
                    help="max number of covers to use (default: %(default)s; 0 = all)")
    ap.add_argument("--ratio", type=float, default=None,
                    help="run ONLY this impact ratio on all covers and print paper-ready numbers "
                         "(e.g. --ratio 0.25)")
    ap.add_argument("--force", action="store_true", help="ignore cached per-cover results")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip the sweep; analyse and plot from the saved CSV")
    args = ap.parse_args()

    if args.ratio is not None:
        run_fixed_ratio(args.ratio, args.max_covers or None,
                        force=args.force, analyze_only=args.analyze_only)
        print("\n" + "=" * 85)
        print(f"🎉 SUCCESS! Fixed-ratio run (r = {args.ratio:g}) completed.")
        print("=" * 85)
        return

    if args.analyze_only:
        if not os.path.exists(OUTPUT_CSV):
            print(f"❌ {OUTPUT_CSV} not found; run the sweep first.")
            return
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = run_sweep(args.max_covers or None, force=args.force)

    if df is None:
        return
    analyze(df)

    print("\n" + "=" * 85)
    print("🎉 SUCCESS! sweeping.py completed.")
    print(f"• Data:    {OUTPUT_CSV}")
    print(f"• Summary: {OUTPUT_SUMMARY}")
    print(f"• Figures: {OUTPUT_GRAPH}")
    print(f"          {OUTPUT_KNEE_GRAPH}")
    print("=" * 85)


if __name__ == "__main__":
    main()