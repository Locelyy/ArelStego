"""
make_histogram_stego.py -- full pipeline (GA + Shamir, n = 4 shares) at ratio 0.25, 100 Kb payload,
on the 15 cover images, saving cover and all shares as lossless TIFF for the MATLAB histogram analysis.

Run from the same folder as GAPS.py, Cover/ and payload/:
    python make_histogram_stego.py

Output (HistogramStego/):
    cover/image_1.tiff ...
    stego/image_1_share1.tiff ... image_1_share4.tiff ...
    histogram_stego_metrics.csv     (GA/Shamir bits, PSNR, SSIM for every share)
"""
import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd
import cv2

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
    SHAMIR_THRESHOLD,
)

COVER_PATH = os.path.join(BASE_PATH, "Cover")
PAYLOAD_PATH = os.path.join(BASE_PATH, "payload")
OUT = os.path.join(BASE_PATH, "HistogramStego")

RATIO = 0.25
PAYLOAD_KB = 100
N_SHARES = 4                      # fixed to 4 for this experiment
COVER_FILES = [f"image {i}.tiff" for i in range(1, 16)]


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def main():
    os.makedirs(os.path.join(OUT, "cover"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "stego"), exist_ok=True)

    p_path = os.path.join(PAYLOAD_PATH, f"random-binary_{PAYLOAD_KB}Kb.txt")
    if not os.path.exists(p_path):
        sys.exit(f"Payload not found: {p_path}")
    with open(p_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    with quiet():
        bits, _ = PayloadProcessor().process_for_embedding(txt)
    total = len(bits)
    print(f"Payload {PAYLOAD_KB} Kb -> {total} bits | ratio {RATIO} | shares n = {N_SHARES}, k = {SHAMIR_THRESHOLD}")

    rows = []
    for cname in COVER_FILES:
        stem = os.path.splitext(cname)[0].replace(" ", "_")
        cover = load_image_safe(os.path.join(COVER_PATH, cname))
        if cover is None:
            print(f"[skip] cannot load {cname}")
            continue
        try:
            with quiet():
                stego_ga, idx_ga, _, bits_in_ga = embed_group_average_grey(
                    cover, bits, use_impact_analysis=True, impact_pool_ratio=RATIO)
                remaining = total - bits_in_ga
                if remaining > 0:
                    shares, _ = embed_shamir_n_stego(
                        stego_ga, bits[bits_in_ga:], N_SHARES, SHAMIR_THRESHOLD,
                        exclusion_indices=idx_ga)
                else:
                    shares = [stego_ga.copy() for _ in range(N_SHARES)]
        except Exception as e:
            print(f"[fail] {cname}: {type(e).__name__}: {e}")
            continue

        cv2.imwrite(os.path.join(OUT, "cover", f"{stem}.tiff"), np.asarray(cover).astype(np.uint8))
        for k, s in enumerate(shares, 1):
            s8 = np.asarray(s).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, "stego", f"{stem}_share{k}.tiff"), s8)
            with quiet():
                m = calculate_metrics(f"{stem}_s{k}", cover, s8, total)
            rows.append({"Cover": stem, "Share": k, "Total Bits": total, "GA Bits": bits_in_ga,
                         "Shamir Bits": max(0, remaining),
                         "GA Share (%)": 100.0 * bits_in_ga / total,
                         "PSNR (dB)": m["psnr"], "SSIM": m["ssim"]})
        print(f"{cname}: GA {bits_in_ga:,} bits ({100.0 * bits_in_ga / total:.1f}%), "
              f"Shamir {max(0, remaining):,} bits | share-1 PSNR {rows[-N_SHARES]['PSNR (dB)']:.2f} dB")

    if not rows:
        sys.exit("Nothing was produced.")
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT, "histogram_stego_metrics.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved {df['Cover'].nunique()} covers x {N_SHARES} shares to {OUT}")
    print(f"Metrics: {csv}")
    print(f"Mean PSNR {df['PSNR (dB)'].mean():.2f} dB | mean SSIM {df['SSIM'].mean():.5f} | "
          f"mean GA share {df['GA Share (%)'].mean():.1f}%")


if __name__ == "__main__":
    main()