"""
bossbase_embed_flat.py -- build a cover/stego dataset from BOSSbase for steganalysis.
Flat layout (no train/val/test folders):

    <out>/cover/NAME.pgm
    <out>/stego/r<ratio>_p<kb>Kb/NAME.pgm

Run it from the same folder as GAPS.py and the payload/ folder (same layout as sweeping.py).

What it does
  1. Picks N random BOSSbase covers (fixed seed).
  2. Copies the chosen covers to   <out>/cover/NAME.pgm
  3. Runs GA + Shamir embedding for every (ratio, payload) combination and saves
     share 0 as                    <out>/stego/r<ratio>_p<kb>Kb/NAME.pgm
     (use --all-shares to also save the other shares in ..._share<k> folders).
  4. Saves split.csv (a suggested train/val/test assignment per cover, NOT used for folders)
     and metrics_*.csv (PSNR/SSIM per image).

Examples
  python bossbase_embed_flat.py --bossbase D:/BOSSbase_1.01 --n 20 --ratios 0.25 --payloads 10      # pilot
  python bossbase_embed_flat.py --bossbase D:/BOSSbase_1.01 --n 5000 --ratios 0.07 0.25 --payloads 10 50 100 --workers 8

Re-running resumes: images whose stego file already exists are skipped.
"""
import os
import sys
import io
import glob
import shutil
import argparse
import contextlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

from GAPS import (
    load_image_safe,
    embed_group_average_grey,
    embed_shamir_n_stego,
    calculate_metrics,
    PayloadProcessor,
    NUM_STEGO_IMAGES,
    SHAMIR_THRESHOLD,
)

PAYLOAD_PATH = os.path.join(BASE_PATH, "payload")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------- payloads
_payload_cache = {}


def get_payload_bits(kb):
    """Same loading path as sweeping.py, so the bit format is guaranteed to be accepted."""
    if kb not in _payload_cache:
        p = os.path.join(PAYLOAD_PATH, f"random-binary_{kb}Kb.txt")
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        with quiet():
            bits, _ = PayloadProcessor().process_for_embedding(txt)
        _payload_cache[kb] = bits
    return _payload_cache[kb]


def randomize_like(bits, rng):
    """Fresh random bits with the same length and container type as `bits`.
    Different payload per cover, so the detector cannot learn one fixed message."""
    n = len(bits)
    r = rng.integers(0, 2, n)
    if isinstance(bits, str):
        return "".join("1" if b else "0" for b in r)
    if isinstance(bits, np.ndarray):
        return r.astype(bits.dtype)
    if n and isinstance(bits[0], str):
        return ["1" if b else "0" for b in r]
    return [int(b) for b in r]


# ---------------------------------------------------------------- worker
def embed_job(job):
    path, split, ratio, kb, out, idx, seed, rand, all_shares = job
    name = os.path.splitext(os.path.basename(path))[0]
    tag = f"r{ratio:g}_p{kb}Kb"
    first = os.path.join(out, "stego", f"{name}.pgm")
    base = {"Cover": name, "Split": split, "Ratio": ratio, "Payload (KB)": kb}
    if os.path.exists(first):
        return {**base, "Status": "skipped"}
    try:
        cover = load_image_safe(path)
        bits = get_payload_bits(kb)
        if rand:
            bits = randomize_like(bits, np.random.default_rng(seed * 100003 + idx))
        total = len(bits)
        with quiet():
            stego_ga, indices_ga, _, bits_in_ga = embed_group_average_grey(
                cover, bits, use_impact_analysis=True, impact_pool_ratio=ratio)
            remaining = total - bits_in_ga
            if remaining > 0:
                shares, _ = embed_shamir_n_stego(
                    stego_ga, bits[bits_in_ga:], NUM_STEGO_IMAGES, SHAMIR_THRESHOLD,
                    exclusion_indices=indices_ga)
            else:
                shares = [stego_ga.copy() for _ in range(NUM_STEGO_IMAGES)]
            m = calculate_metrics("bb", cover, shares[0], total)

        for k, s in enumerate(shares if all_shares else shares[:1]):
            d = os.path.join(out, "stego" if k == 0 else f"stego_share{k}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(np.asarray(s).astype(np.uint8)).save(os.path.join(d, f"{name}.pgm"))

        return {**base, "Status": "ok", "Total Bits": total, "GA Bits": bits_in_ga,
                "Shamir Bits": max(0, remaining), "PSNR (dB)": m["psnr"], "SSIM": m["ssim"]}
    except Exception as e:
        return {**base, "Status": f"failed: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bossbase", required=True, help="folder with the BOSSbase .pgm files")
    ap.add_argument("--out", default=os.path.join(BASE_PATH, "steg_dataset"))
    ap.add_argument("--n", type=int, default=5000, help="number of covers to use")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.07, 0.25])
    ap.add_argument("--payloads", type=int, nargs="+", default=[10, 50, 100], help="payload sizes in KB")
    ap.add_argument("--split", type=float, nargs=3, default=[0.6, 0.2, 0.2], metavar=("TRAIN", "VAL", "TEST"),
                    help="only used for the suggested assignment in split.csv; folders are NOT split")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-random-payload", action="store_true",
                    help="use the same payload file content for every cover (default: fresh random bits per cover)")
    ap.add_argument("--all-shares", action="store_true", help="also save shares 1..n-1")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.bossbase, "*.pgm")))
    if not files:
        sys.exit(f"No .pgm files found in {args.bossbase}")
    if args.n > len(files):
        sys.exit(f"Requested {args.n} covers but only {len(files)} found.")
    for kb in args.payloads:
        if not os.path.exists(os.path.join(PAYLOAD_PATH, f"random-binary_{kb}Kb.txt")):
            sys.exit(f"Missing payload file for {kb} KB in {PAYLOAD_PATH}")

    if len(args.ratios) * len(args.payloads) != 1:
        sys.exit("Stego images are saved directly in stego/, so run exactly one ratio and one payload "
                 "per output folder (e.g. --ratios 0.25 --payloads 100). Use a different --out for another setting.")

    # ---- 1. sample covers + suggested split (csv only)
    rng = np.random.default_rng(args.seed)
    chosen = [files[i] for i in rng.permutation(len(files))[:args.n]]
    n_tr = int(round(args.split[0] * args.n))
    n_va = int(round(args.split[1] * args.n))
    splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * (args.n - n_tr - n_va)
    split_df = pd.DataFrame({"Cover": [os.path.splitext(os.path.basename(p))[0] for p in chosen],
                             "Split": splits, "Path": chosen})
    os.makedirs(args.out, exist_ok=True)
    split_df.to_csv(os.path.join(args.out, "split.csv"), index=False)
    print(f"Covers: {args.n} | suggested train/val/test = {n_tr}/{n_va}/{args.n - n_tr - n_va} "
          f"(in split.csv only) | seed {args.seed}")

    # ---- 2. copy covers (flat)
    cover_dir = os.path.join(args.out, "cover")
    os.makedirs(cover_dir, exist_ok=True)
    for p in chosen:
        dst = os.path.join(cover_dir, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)

    # ---- 3. embedding
    for ratio in args.ratios:
        for kb in args.payloads:
            tag = f"r{ratio:g}_p{kb}Kb"
            print(f"\n=== {tag} ===")
            jobs = [(p, s, ratio, kb, args.out, i, args.seed,
                     not args.no_random_payload, args.all_shares)
                    for i, (p, s) in enumerate(zip(chosen, splits))]
            rows = []
            with Pool(args.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(embed_job, jobs, chunksize=4), 1):
                    rows.append(r)
                    if i % 50 == 0 or i == len(jobs):
                        print(f"  {i}/{len(jobs)}", flush=True)

            new = pd.DataFrame(rows)
            mpath = os.path.join(args.out, f"metrics_{tag}.csv")
            ok = new[new["Status"] == "ok"]
            if os.path.exists(mpath):
                ok = pd.concat([pd.read_csv(mpath), ok]).drop_duplicates("Cover", keep="last")
            ok.to_csv(mpath, index=False)
            failed = new[new["Status"].str.startswith("failed")]
            if len(failed):
                failed.to_csv(os.path.join(args.out, f"failed_{tag}.csv"), index=False)
                print(f"  {len(failed)} failed (see failed_{tag}.csv). First: {failed['Status'].iloc[0]}")
            if len(ok):
                print(f"  PSNR {ok['PSNR (dB)'].mean():.2f} ± {ok['PSNR (dB)'].std():.2f} dB | "
                      f"SSIM {ok['SSIM'].mean():.5f} ± {ok['SSIM'].std():.5f} | "
                      f"GA share {100 * (ok['GA Bits'] / ok['Total Bits']).mean():.1f}%")

    print(f"\nDone. Dataset in: {args.out}")
    print("  cover/*.pgm   stego/*.pgm   split.csv   metrics_*.csv")


if __name__ == "__main__":
    main()