# ablation2.py
# Ablation study for ArelStego (Stage 1 adaptive group-average + Stage 2 Shamir overflow).
# Must be placed in the same folder as GAPS.py (your main script).
#
# Studies:
#   A. Components (impact ratio fixed at --ratio, default 0.25):
#        full, no_ordering, no_impact_filter, neither, reverse_ordering,
#        inverted_impact, random_pool, plain_lsb (baseline)
#   B. Impact-ratio sweep, with and without ordering.
#
# Stage 2 modes:
#   default        : simulated (random LSB bits at the real Shamir positions) -> fast
#   --real-stage2  : runs YOUR Shamir code from GAPS.py and verifies payload + cover recovery
#
# Outputs (in --out-dir): ablation_raw.csv, ablation_summary.csv, PNG plots
#   (ablation_components.png, ablation_ratio.png, ablation_bars.png).

import argparse
import contextlib
import io
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import uniform_filter, sobel, generic_filter
from skimage.metrics import peak_signal_noise_ratio as psnr

from GAPS import (
    load_image_safe,
    embed_group_average_grey,
    extract_group_average_grey,
    embed_shamir_n_stego,
    extract_shamir_n_stego,
    calculate_metrics,
    PayloadProcessor,
    NUM_STEGO_IMAGES,
    SHAMIR_THRESHOLD,
)

# ==========================================
# CONFIG
# ==========================================
COVER_PATH = "/home/marin/Documents/AntiGravity/Cover"
PAYLOAD_PATH = "/home/marin/Documents/AntiGravity/payload"
OUT_PATH = "/home/marin/Documents/AntiGravity/Results/Ablation"

COVER_FILES = [f"image {i}.tiff" for i in range(1, 16)]
PAYLOAD_SIZES = [1] + list(range(10, 101, 10))
PAYLOAD_TEMPLATE = "random-binary_{}Kb.txt"

GROUP_SIZE = 5
RATIOS = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SHUFFLE_SEED = 999          # same seed your Stage 2 uses
BASE_SEED = 12345           # seed for the stochastic ablation variants

# Grouped bar chart: legend label -> variant name. EDIT to match your paper's naming.
BAR_VARIANTS = {
    "Ordering":          "full",
    "No Ordering": "no_ordering",
}
BAR_COLORS = ["#5E8DB8", "#FF9A1F"]


def quiet(fn, *a, **kw):
    """Call fn with its print output suppressed (your GAPS functions print a lot)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ==========================================
# VARIANT DEFINITIONS
# pool  : top | bottom | random | all     (how the safe pool is chosen)
# order : ascending | descending | random (order in which pool pixels receive bits)
# kind  : group_avg | lsb
# ==========================================
def build_variants(ratio):
    v = {
        "full":             dict(group="component", kind="group_avg", pool="top",    ratio=ratio, order="ascending"),
        "no_ordering":      dict(group="component", kind="group_avg", pool="top",    ratio=ratio, order="random"),
        "no_impact_filter": dict(group="component", kind="group_avg", pool="all",    ratio=1.0,   order="ascending"),
        "neither":          dict(group="component", kind="group_avg", pool="all",    ratio=1.0,   order="random"),
        "reverse_ordering": dict(group="component", kind="group_avg", pool="top",    ratio=ratio, order="descending"),
        "inverted_impact":  dict(group="component", kind="group_avg", pool="bottom", ratio=ratio, order="ascending"),
        "random_pool":      dict(group="component", kind="group_avg", pool="random", ratio=ratio, order="ascending"),
        "plain_lsb":        dict(group="component", kind="lsb",       pool="all",    ratio=1.0,   order="random"),
    }
    for r in RATIOS:
        v[f"ordered_r{r:.2f}"] = dict(group="ratio_ordered", kind="group_avg", pool="top", ratio=r, order="ascending")
        v[f"unordered_r{r:.2f}"] = dict(group="ratio_unordered", kind="group_avg", pool="top", ratio=r, order="random")
    return v


def is_stochastic(cfg):
    return cfg["order"] == "random" or cfg["pool"] == "random" or cfg["kind"] == "lsb"


# ==========================================
# UTILITIES
# ==========================================
def load_payload_bits(path):
    """Uses your PayloadProcessor (clean 0/1, pad to bytes, 32-bit length header)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    proc = PayloadProcessor()
    bit_str, _ = quiet(proc.process_for_embedding, text)
    return np.frombuffer(bit_str.encode(), dtype=np.uint8).astype(np.int64) - 48


def random_payload_bits(size_kb):
    data = np.random.RandomState(size_kb).randint(0, 2, size_kb * 1000).astype(np.int64)
    header = np.array(list(format(len(data), "032b")), dtype=np.int64)
    return np.concatenate([header, data])


def group_average_table(p, group_size=GROUP_SIZE):
    """Per-pixel group average (identical to embed_group_average_grey)."""
    n = group_size
    edges = np.arange(0, 256 + n, n)
    count, _ = np.histogram(p, bins=edges)
    count[count == 0] = 1
    sum_each, _ = np.histogram(p, bins=np.arange(257))
    sum_group = np.zeros_like(count, dtype=np.int64)
    for i in range(256):
        sum_group[i // n] += sum_each[i] * i
    group_avg_map = np.ceil(sum_group / count).astype(np.int64)
    return group_avg_map[p // n]


def impact_map(channel, exact=False):
    """Local variance (3x3) / max + Sobel magnitude / max.
    Fast path uses E[x^2]-E[x]^2 (equivalent to generic_filter(np.var) up to float error)."""
    x = channel.astype(np.float64)
    if exact:
        var = generic_filter(x, np.var, size=3, mode="reflect")
    else:
        mean = uniform_filter(x, size=3, mode="reflect")
        mean_sq = uniform_filter(x * x, size=3, mode="reflect")
        var = np.maximum(mean_sq - mean ** 2, 0.0)
    edge = np.sqrt(sobel(x, axis=0) ** 2 + sobel(x, axis=1) ** 2)
    return var / (var.max() + 1e-10) + edge / (edge.max() + 1e-10)


def build_context(cover, exact_impact):
    p = cover.flatten().astype(np.int64)
    avg = group_average_table(p)
    cand = np.where((p >= 5) & (p <= 249))[0]
    t0 = time.perf_counter()
    impact = impact_map(cover, exact_impact).flatten()
    t_impact = time.perf_counter() - t0
    return dict(
        p=p, avg=avg, cand=cand, impact=impact,
        impact_order=np.argsort(impact[cand], kind="stable"),   # ascending impact
        t_impact=t_impact, shape=cover.shape, N=p.size,
    )


def compute_metrics(cover, stego_flat, L):
    """Uses your calculate_metrics from GAPS."""
    stego = stego_flat.reshape(cover.shape).astype(np.uint8)
    m = quiet(calculate_metrics, "ablation", cover, stego, L)
    return float(m["psnr"]), float(m["mse"]), float(m["ssim"]), float(m["bpp"])


def psnr_only(cover, stego_img):
    ps = psnr(cover, stego_img.astype(np.uint8), data_range=255)
    return 100.0 if np.isinf(ps) else float(ps)


# ==========================================
# REAL STAGE 2 + END-TO-END VERIFICATION (uses your GAPS functions)
# ==========================================
def real_stage2_embed(cover, stego_flat, idx, overflow_bits, py_seed):
    random.seed(py_seed)        # reproducible random polynomial coefficients
    stego2d = stego_flat.reshape(cover.shape).astype(np.uint8)
    bit_str = "".join(map(str, overflow_bits.tolist()))
    return quiet(embed_shamir_n_stego, stego2d, bit_str, NUM_STEGO_IMAGES,
                 threshold=SHAMIR_THRESHOLD, exclusion_indices=idx)


def verify_pipeline(cover, first_stego_flat, imgs, orig_lsbs, idx, avg, bits, nb):
    """Receiver side: extract Stage 1 bits, extract Stage 2 bits (images 1..t), rebuild cover."""
    L = len(bits)
    s1 = first_stego_flat.astype(np.int64)
    rec = s1.copy()

    if nb > 0:
        s = s1[idx]
        ga_bits = (s + avg[idx]) % 2
        rec[idx] = (s + avg[idx] - ga_bits) // 2
    else:
        ga_bits = np.zeros(0, dtype=np.int64)

    overflow = L - nb
    if overflow > 0:
        sh_str = quiet(extract_shamir_n_stego, imgs, overflow,
                       threshold=SHAMIR_THRESHOLD, exclusion_indices=idx)
        sh_bits = np.frombuffer(sh_str.encode(), dtype=np.uint8).astype(np.int64) - 48
        for k, v in orig_lsbs.items():
            rec[k] = (rec[k] & 0xFE) | v
        all_bits = np.concatenate([ga_bits, sh_bits])
    else:
        all_bits = ga_bits

    payload_ok = (len(all_bits) == L) and np.array_equal(all_bits, bits)
    cover_ok = np.array_equal(rec.reshape(cover.shape).astype(np.uint8), cover)
    return payload_ok, cover_ok


# ==========================================
# CONSISTENCY CHECK: my 'full' variant vs your real embed_group_average_grey
# ==========================================
def check_against_real(cover, bits_by_size, ratio):
    print("\n=== Consistency check: ablation 'full' vs GAPS.embed_group_average_grey ===")
    print("(uses the exact generic_filter impact map; exact pixel-for-pixel equality is NOT expected")
    print(" because ties in the sorts can be broken differently, but bit counts and PSNR should match)")
    ctx = build_context(cover, exact=True) if False else build_context(cover, True)
    cfg = build_variants(ratio)["full"]
    sizes = sorted(bits_by_size)
    for size in [sizes[0], sizes[-1]]:
        bits = bits_by_size[size]
        bit_str = "".join(map(str, bits.tolist()))
        stego_real, idx_real, avg_real, nb_real = quiet(
            embed_group_average_grey, cover, bit_str,
            use_impact_analysis=True, impact_pool_ratio=ratio)
        mine = run_variant(cover, ctx, bits, cfg, np.random.default_rng(0), stage2="sim")

        rec, ext = quiet(extract_group_average_grey, stego_real, idx_real, avg_real)
        ext_ok = (ext == bit_str[:nb_real]) and np.array_equal(rec, cover)

        line = (f"  {size:>3} Kb | bits in Stage 1: real={nb_real} mine={mine['bits_ga']} "
                f"-> {'MATCH' if nb_real == mine['bits_ga'] else 'MISMATCH'} | "
                f"real extractor restores cover+bits: {ext_ok}")
        if nb_real == len(bits):          # no overflow -> PSNR is directly comparable
            m_real = quiet(calculate_metrics, "real", cover, stego_real, len(bits))
            gap = mine["psnr"] - float(m_real["psnr"])
            line += f" | PSNR gap (mine-real): {gap:+.4f} dB"
        print(line)


# ==========================================
# CORE: ONE ABLATION RUN
# ==========================================
def run_variant(cover, ctx, bits, cfg, rng, stage2="sim"):
    """stage2: 'sim' (random LSB bits) or 'real' (your Shamir code + full verification)."""
    p, avg, cand, N = ctx["p"], ctx["avg"], ctx["cand"], ctx["N"]
    L = len(bits)
    row = dict(L=L, N=N, status="ok", bits_ga=0, bits_overflow=0,
               frac_d0=np.nan, mean_abs_d=np.nan, ga_reversible=np.nan,
               stage2_mode="none", is_correct=np.nan, cover_recovered=np.nan,
               psnr_all4=np.nan, verify_time_s=np.nan)

    t0 = time.perf_counter()
    stego = p.copy()

    # ---------- baseline: plain LSB replacement ----------
    if cfg["kind"] == "lsb":
        if L > N:
            row["status"] = "capacity_exceeded"
            return row
        sel = rng.permutation(N)[:L]
        stego[sel] = (stego[sel] & 0xFE) | bits
        row["bits_overflow"] = L
        row["embed_time_s"] = time.perf_counter() - t0
        row["psnr"], row["mse"], row["ssim"], row["bpp"] = compute_metrics(cover, stego, L)
        return row

    # ---------- pool selection ----------
    n_c = len(cand)
    if cfg["pool"] == "all":
        pool = cand
    else:
        k = int(n_c * cfg["ratio"])
        if k <= 0:
            pool = cand[:0]
        elif cfg["pool"] == "top":
            pool = cand[ctx["impact_order"][-k:]]
        elif cfg["pool"] == "bottom":
            pool = cand[ctx["impact_order"][:k]]
        elif cfg["pool"] == "random":
            pool = rng.choice(cand, size=k, replace=False)
        else:
            raise ValueError(cfg["pool"])

    # ---------- ordering inside the pool ----------
    d = p[pool] - avg[pool]
    if cfg["order"] == "ascending":
        o = np.argsort(np.abs(d), kind="stable")
    elif cfg["order"] == "descending":
        o = np.argsort(-np.abs(d), kind="stable")
    elif cfg["order"] == "random":
        o = rng.permutation(len(pool))
    else:
        raise ValueError(cfg["order"])
    final, d_sorted = pool[o], d[o]

    # ---------- Stage 1 embedding: new = 2*val - avg + bit ----------
    nb = min(L, len(final))
    idx = final[:nb]
    new_vals = 2 * p[idx] - avg[idx] + bits[:nb]
    if nb > 0 and (new_vals.min() < 0 or new_vals.max() > 255):
        row["status"] = "overflow_error"
        return row
    stego[idx] = new_vals
    row["bits_ga"] = nb
    if nb > 0:
        row["frac_d0"] = float(np.mean(d_sorted[:nb] == 0))
        row["mean_abs_d"] = float(np.mean(np.abs(d_sorted[:nb])))

    # ---------- Stage 1 reversibility check ----------
    if nb > 0:
        s = stego[idx]
        bit_rec = (s + avg[idx]) % 2
        orig_rec = (s + avg[idx] - bit_rec) // 2
        row["ga_reversible"] = float(np.array_equal(bit_rec, bits[:nb]) and np.array_equal(orig_rec, p[idx]))
    else:
        row["ga_reversible"] = 1.0

    overflow = L - nb
    row["bits_overflow"] = overflow
    if overflow > N - nb:
        row["status"] = "capacity_exceeded"
        return row

    imgs, orig_lsbs, first_flat = None, {}, stego

    # ---------- Stage 2 ----------
    if overflow > 0 and stage2 == "real":
        py_seed = int(rng.integers(0, 2 ** 31 - 1))
        imgs, orig_lsbs = real_stage2_embed(cover, stego, idx, bits[nb:], py_seed)
        if imgs is None:
            row["status"] = "capacity_exceeded"
            return row
        first_flat = imgs[0].flatten()
        row["stage2_mode"] = "real"
    elif overflow > 0:
        # Simulated: share bytes are ~uniform, so their LSBs are ~fair coin flips.
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        avail = np.random.RandomState(SHUFFLE_SEED).permutation(np.where(mask)[0])
        sel = avail[:overflow]
        share_bits = rng.integers(0, 2, size=overflow)
        stego[sel] = (stego[sel] & 0xFE) | share_bits
        row["stage2_mode"] = "sim"

    row["embed_time_s"] = time.perf_counter() - t0
    row["psnr"], row["mse"], row["ssim"], row["bpp"] = compute_metrics(cover, first_flat, L)

    # ---------- full-pipeline verification (real mode only) ----------
    if stage2 == "real":
        t1 = time.perf_counter()
        if imgs is not None:
            row["psnr_all4"] = float(np.mean([psnr_only(cover, im) for im in imgs]))
        else:
            row["psnr_all4"] = row["psnr"]          # Stage-1-only: the 4 images are identical copies
        payload_ok, cover_ok = verify_pipeline(cover, first_flat, imgs, orig_lsbs, idx, avg, bits, nb)
        row["is_correct"] = float(payload_ok)
        row["cover_recovered"] = float(cover_ok)
        row["verify_time_s"] = time.perf_counter() - t1
    return row


# ==========================================
# PLOTTING
# ==========================================
def plot_components(summary, out_dir):
    df = summary[summary["group"] == "component"]
    plt.figure(figsize=(8, 5))
    for name, g in df.groupby("variant"):
        g = g.sort_values("payload_kb")
        plt.errorbar(g["payload_kb"], g["psnr_mean"], yerr=g["psnr_std"], marker="o",
                     capsize=2, label=name)
    plt.xlabel("Payload size (Kb)")
    plt.ylabel("PSNR (dB)")
    plt.title("Component ablation (stego image 1)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_components.png"), dpi=200)
    plt.close()


def plot_ratio(summary, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, grp, title in zip(axes, ["ratio_ordered", "ratio_unordered"],
                              ["With ordering", "Without ordering (random)"]):
        df = summary[summary["group"] == grp]
        for kb, g in df.groupby("payload_kb"):
            g = g.sort_values("ratio")
            ax.plot(g["ratio"], g["psnr_mean"], marker="o", label=f"{kb} Kb")
        ax.set_xlabel("IMPACT_RATIO")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].legend(fontsize=7, title="Payload", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_ratio.png"), dpi=200)
    plt.close()


def plot_bars(summary, out_dir, sizes=(60, 70, 80, 90, 100), mapping=None,
              decimals=4, fname="ablation_bars.png"):
    """Grouped bar chart: one cluster per payload size, one bar per variant,
    rotated value labels, zoomed y-axis with a break marker."""
    mapping = mapping or BAR_VARIANTS
    df = summary[summary["group"] == "component"]
    piv = df.pivot(index="variant", columns="payload_kb", values="psnr_mean")

    sizes = [s for s in sizes if s in piv.columns]
    missing = [v for v in mapping.values() if v not in piv.index]
    if not sizes or missing:
        print(f"[!] plot_bars skipped (sizes found: {sizes}, missing variants: {missing})")
        return

    vals = np.array([[piv.loc[v, s] for s in sizes] for v in mapping.values()])  # (n_var, n_size)
    n_var, n_size = vals.shape
    width = 0.2 if n_var == 4 else 0.8 / n_var
    x = np.arange(n_size)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color) in enumerate(zip(mapping.keys(), BAR_COLORS)):
        pos = x + (i - (n_var - 1) / 2) * width
        bars = ax.bar(pos, vals[i], width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals[i]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.{decimals}f}",
                    ha="center", va="bottom", rotation=90, fontsize=8)

    # Zoomed y-axis with headroom for the rotated labels
        vmin, vmax = vals.min(), vals.max()
    if vmax - vmin > 10:            # wide range: ticks every 5 dB
        step = 5
        lo = np.floor((vmin - 0.3) / step) * step
        hi = vmax + 0.12 * (vmax - lo)
    else:                           # narrow range: fine ticks as before
        step = 0.5
        lo = np.floor((vmin - 0.3) * 2) / 2
        hi = vmax + 0.9
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}Kb" for s in sizes])
    ax.set_xlabel("Payload Size")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    # Axis-break marker (the "//" at the bottom of the y-axis)
    kw = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
    ax.plot((-0.012, 0.012), (-0.02, 0.02), **kw)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover-dir", default=COVER_PATH)
    ap.add_argument("--payload-dir", default=PAYLOAD_PATH)
    ap.add_argument("--out-dir", default=OUT_PATH)
    ap.add_argument("--ratio", type=float, default=0.25, help="IMPACT_RATIO for the component ablation")
    ap.add_argument("--repeats", type=int, default=3, help="repeats for stochastic variants")
    ap.add_argument("--exact-impact", action="store_true",
                    help="use the original generic_filter impact map (slow)")
    ap.add_argument("--quick", action="store_true", help="3 covers, payloads 1/10/50/100 Kb")
    ap.add_argument("--random-payload", action="store_true",
                    help="if a payload file is missing, generate random bits (testing only)")
    ap.add_argument("--real-stage2", action="store_true",
                    help="run your actual Shamir stage and verify payload + cover recovery (slow)")
    ap.add_argument("--real-variants", default="all",
                    help="comma-separated variant names that use the real Stage 2 (default: all). "
                         "Others stay simulated. Example: full,no_ordering,ordered_r0.05")
    ap.add_argument("--compare-sim", action="store_true",
                    help="with --real-stage2: also run the simulated Stage 2 on the same config "
                         "and record the PSNR gap (validates the simulation)")
    ap.add_argument("--check-real", action="store_true",
                    help="before the study, compare the ablation 'full' variant with "
                         "GAPS.embed_group_average_grey on the first cover (slow: exact impact map)")
    ap.add_argument("--bar-sizes", default="60,70,80,90,100",
                    help="payload sizes (Kb) shown in the grouped bar chart")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    covers = COVER_FILES[:3] if args.quick else COVER_FILES
    sizes = [1, 10, 50, 100] if args.quick else PAYLOAD_SIZES
    variants = build_variants(args.ratio)

    if args.real_variants == "all":
        real_set = set(variants)
    else:
        real_set = {v.strip() for v in args.real_variants.split(",") if v.strip()}
        unknown = real_set - set(variants)
        if unknown:
            raise SystemExit(f"Unknown variant(s) in --real-variants: {sorted(unknown)}\n"
                             f"Valid: {sorted(variants)}")

    # ---- load payloads once ----
    payloads = {}
    for size in sizes:
        ppath = os.path.join(args.payload_dir, PAYLOAD_TEMPLATE.format(size))
        if os.path.exists(ppath):
            payloads[size] = load_payload_bits(ppath)
        elif args.random_payload:
            payloads[size] = random_payload_bits(size)
        else:
            print(f"[!] Missing payload {size}Kb")

    if args.check_real and payloads:
        first = None
        for cname in covers:
            first = quiet(load_image_safe, os.path.join(args.cover_dir, cname))
            if first is not None:
                break
        if first is not None:
            check_against_real(first, payloads, args.ratio)

    rows = []
    for ci, cname in enumerate(covers):
        cover = quiet(load_image_safe, os.path.join(args.cover_dir, cname))
        if cover is None:
            print(f"[!] Could not load {cname}")
            continue
        print(f"\n[{ci + 1}/{len(covers)}] {cname}  shape={cover.shape}")
        ctx = build_context(cover, args.exact_impact)

        for pi, size in enumerate(sizes):
            if size not in payloads:
                continue
            bits = payloads[size]

            for vname, cfg in variants.items():
                use_real = args.real_stage2 and cfg["kind"] != "lsb" and vname in real_set
                n_rep = args.repeats if is_stochastic(cfg) else 1
                for rep in range(n_rep):
                    seed = [BASE_SEED, ci, pi, rep]
                    r = run_variant(cover, ctx, bits, cfg, np.random.default_rng(seed),
                                    stage2="real" if use_real else "sim")
                    r["psnr_sim"] = np.nan
                    if use_real and args.compare_sim and r["status"] == "ok":
                        rs = run_variant(cover, ctx, bits, cfg, np.random.default_rng(seed), stage2="sim")
                        if rs["status"] == "ok":
                            r["psnr_sim"] = rs["psnr"]
                    r.update(cover=cname, payload_kb=size, variant=vname, group=cfg["group"],
                             pool=cfg["pool"], order=cfg["order"], ratio=cfg["ratio"],
                             repeat=rep, impact_time_s=ctx["t_impact"])
                    rows.append(r)
            print(f"  {size:>3} Kb done")

        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "ablation_raw.csv"), index=False)  # checkpoint

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise SystemExit("No results produced (check cover/payload paths).")
    raw.to_csv(os.path.join(args.out_dir, "ablation_raw.csv"), index=False)

    ok = raw[raw["status"] == "ok"].copy()
    failed = raw[raw["status"] != "ok"]
    if len(failed):
        print(f"\n{len(failed)} runs skipped (capacity exceeded or error); see 'status' in ablation_raw.csv")

    # Paired difference against the 'full' configuration (same cover, same payload)
    full = (ok[ok["variant"] == "full"].groupby(["cover", "payload_kb"])["psnr"].mean()
            .rename("psnr_full"))
    ok = ok.join(full, on=["cover", "payload_kb"])
    ok["delta_psnr_vs_full"] = ok["psnr"] - ok["psnr_full"]
    ok["psnr_gap_real_minus_sim"] = ok["psnr"] - ok["psnr_sim"]

    summary = (ok.groupby(["group", "variant", "payload_kb"], as_index=False)
               .agg(ratio=("ratio", "first"),
                    psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
                    mse_mean=("mse", "mean"), ssim_mean=("ssim", "mean"),
                    delta_vs_full=("delta_psnr_vs_full", "mean"),
                    bits_ga=("bits_ga", "mean"), bits_overflow=("bits_overflow", "mean"),
                    frac_d0=("frac_d0", "mean"), mean_abs_d=("mean_abs_d", "mean"),
                    ga_reversible=("ga_reversible", "mean"),
                    stage2_mode=("stage2_mode", "first"),
                    is_correct=("is_correct", "mean"),
                    cover_recovered=("cover_recovered", "mean"),
                    psnr_all4=("psnr_all4", "mean"),
                    psnr_gap_real_minus_sim=("psnr_gap_real_minus_sim", "mean"),
                    embed_time_s=("embed_time_s", "mean"),
                    verify_time_s=("verify_time_s", "mean"),
                    n_runs=("psnr", "count")))
    summary.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)

    plot_components(summary, args.out_dir)
    plot_ratio(summary, args.out_dir)
    plot_bars(summary, args.out_dir,
              sizes=[int(s) for s in args.bar_sizes.split(",") if s.strip()])

    pd.set_option("display.width", 200)
    comp = summary[summary["group"] == "component"]
    print("\n=== Mean PSNR (dB), component ablation ===")
    print(comp.pivot(index="variant", columns="payload_kb", values="psnr_mean").round(3))
    print("\n=== Mean PSNR difference vs 'full' (dB) ===")
    print(comp.pivot(index="variant", columns="payload_kb", values="delta_vs_full").round(3))

    if args.real_stage2:
        real = ok[ok["is_correct"].notna()]
        with_s2 = real[real["bits_overflow"] > 0]
        print("\n=== Full-pipeline verification (real mode) ===")
        print(f"Runs verified:                 {len(real)}")
        if len(real):
            print(f"  payload recovered exactly:   {real['is_correct'].mean() * 100:.2f}%")
            print(f"  cover restored exactly:      {real['cover_recovered'].mean() * 100:.2f}%")
        print(f"Runs where Stage 2 was active: {len(with_s2)}")
        if len(with_s2):
            print(f"  payload recovered exactly:   {with_s2['is_correct'].mean() * 100:.2f}%")
            print(f"  cover restored exactly:      {with_s2['cover_recovered'].mean() * 100:.2f}%")
        bad = real[(real["is_correct"] == 0) | (real["cover_recovered"] == 0)]
        if len(bad):
            print("  FAILURES (see ablation_raw.csv):")
            print(bad[["cover", "payload_kb", "variant", "is_correct", "cover_recovered"]].head(20).to_string(index=False))
        if args.compare_sim:
            g = ok["psnr_gap_real_minus_sim"].dropna()
            if len(g):
                print(f"\nReal vs simulated Stage 2 PSNR gap: mean {g.mean():+.4f} dB, "
                      f"max |gap| {g.abs().max():.4f} dB over {len(g)} runs")

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()