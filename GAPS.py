import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import heapq
import collections
import os
import glob
import re
import random
import time
import pandas as pd
from PIL import Image
import struct

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    from google.colab import drive
    print("Google Colab detected.")
except ImportError:
    print("Local environment detected.")

print("Libraries loaded successfully.")



# ==========================================
# BLOCK 3: SHARED UTILITIES
# ==========================================

def bytes_to_bits(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return ''.join(format(byte, '08b') for byte in data)

def bits_to_bytes(bits):
    padding = (8 - len(bits) % 8) % 8
    bits = bits + '0' * padding
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = int(bits[i:i+8], 2)
        byte_array.append(byte)
    return bytes(byte_array)

def dec2bin(n, bits):
    return format(n, f'0{bits}b')

def bin2dec(binary_str):
    return int(binary_str, 2)

def add_length_header(data_bits):
    length = len(data_bits)
    header = format(length, '032b')
    return header + data_bits

def extract_length_header(data_bits):
    if len(data_bits) < 32:
        return ""
    length = int(data_bits[:32], 2)
    if length > len(data_bits) - 32:
        length = len(data_bits) - 32

    payload = data_bits[32:32+length]
    return payload

def load_image_safe(path):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None

    img = cv2.imread(path, -1)

    if img is None:
        try:
            from PIL import Image
            img_pil = Image.open(path)
            img = np.array(img_pil)
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return None

    if img.dtype == np.uint16:
        print("   ℹ️ Detected 16-bit Medical Image. Converting to 8-bit...")
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def calculate_metrics(title, original_img, stego_img, payload_bits):
    original = original_img.astype(np.uint8)
    stego = stego_img.astype(np.uint8)

    mse_val = psnr(original, stego) if psnr(original, stego) != float('inf') else 100
    mse_raw = np.mean((original.astype(float) - stego.astype(float)) ** 2)

    ssim_val = ssim(original, stego, data_range=255)

    total_pixels = original.shape[0] * original.shape[1]
    bpp = payload_bits / total_pixels

    return {
        'title': title,
        'psnr': mse_val,
        'mse': mse_raw,
        'ssim': ssim_val,
        'bpp': bpp
    }

print("Block 3 (Utilities) loaded successfully.")

# ==========================================
# BLOCK 4: STAGE 1 - ADAPTIVE GA EMBEDDING
# ==========================================
from scipy.ndimage import generic_filter, sobel

def calculate_pixel_impact(channel, window_size=3):
    def local_variance(values):
        return np.var(values)

    variance_map = generic_filter(channel.astype(float), local_variance,
                                  size=window_size, mode='reflect')

    sobel_x = sobel(channel.astype(float), axis=0)
    sobel_y = sobel(channel.astype(float), axis=1)
    edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)

    v_max = np.max(variance_map) + 1e-10
    e_max = np.max(edge_strength) + 1e-10

    complexity = (variance_map / v_max) + (edge_strength / e_max)

    return complexity

def embed_group_average_grey(cover_img, payload_bits, use_impact_analysis=True, impact_pool_ratio=0.5):
    channel = cover_img.copy()
    p = channel.flatten().astype(np.int64)
    h, w = channel.shape

    n = 5
    edges = np.arange(0, 256 + n, n)
    count, _ = np.histogram(p, bins=edges)
    count[count == 0] = 1

    sum_each, _ = np.histogram(channel, bins=np.arange(257))
    sum_group = np.zeros_like(count, dtype=np.int64)
    for i in range(256):
        sum_group[i // n] += sum_each[i] * i

    group_avg_map = np.ceil(sum_group / count).astype(np.int64)
    avg_table = np.array([group_avg_map[val // n] for val in p])

    candidate_mask = (p >= 5) & (p <= 249)
    candidate_indices = np.where(candidate_mask)[0]

    if use_impact_analysis:
        print("    [Stage 1] Calculating Pixel Impact Map...")
        impact_map = calculate_pixel_impact(channel)
        impact_flat = impact_map.flatten()

        cand_impacts = impact_flat[candidate_indices]
        target_pool_size = int(len(candidate_indices) * impact_pool_ratio)
        impact_sorted_args = np.argsort(cand_impacts)

        top_impact_args = impact_sorted_args[-target_pool_size:]
        safe_pool_indices = candidate_indices[top_impact_args]

        print(f"    [Stage 1] Impact Filter: {len(candidate_indices)} -> {len(safe_pool_indices)} pixels (Top {impact_pool_ratio*100}%)")

        pool_p = p[safe_pool_indices]
        pool_avg = avg_table[safe_pool_indices]
        pool_diffs = np.abs(pool_p - pool_avg)

        final_sort_order = np.argsort(pool_diffs)
        final_candidates = safe_pool_indices[final_sort_order]

    else:
        cand_diffs = np.abs(p[candidate_indices] - avg_table[candidate_indices])
        sorted_order = np.argsort(cand_diffs)
        final_candidates = candidate_indices[sorted_order]

    actual_capacity = len(final_candidates)
    bits_to_embed = min(len(payload_bits), actual_capacity)

    if bits_to_embed < len(payload_bits):
        print(f"    [Stage 1] ⚠ ADAPTIVE SPLIT TRIGGERED")
        print(f"    - Payload Size: {len(payload_bits)} bits")
        print(f"    - Safe Pool Capacity: {actual_capacity} bits")
        print(f"    - Overflow to Shamir: {len(payload_bits) - bits_to_embed} bits")
    else:
        print(f"    [Stage 1] Full payload fits in GA Safe Pool ({bits_to_embed} bits).")

    stego_flat = p.copy()
    embedded_indices = final_candidates[:bits_to_embed]

    for i in range(bits_to_embed):
        idx = embedded_indices[i]
        val = p[idx]
        avg = avg_table[idx]
        bit = int(payload_bits[i])

        new_val = 2 * val - avg + bit
        stego_flat[idx] = new_val

    stego_img = stego_flat.reshape((h, w)).astype(np.uint8)

    return stego_img, embedded_indices, avg_table, bits_to_embed

def extract_group_average_grey(stego_img, indices, avg_table):
    s = stego_img.flatten().astype(np.int64)
    recovered_flat = s.copy()
    extracted_bits = []

    for idx in indices:
        val = s[idx]
        avg = avg_table[idx]
        bit = (val + avg) % 2
        extracted_bits.append(str(bit))

        orig_p = (val + avg - bit) // 2
        recovered_flat[idx] = orig_p

    recovered_img = recovered_flat.reshape(stego_img.shape).astype(np.uint8)
    return recovered_img, "".join(extracted_bits)

print("Block 4 (Adaptive GA) loaded successfully.")


# ==========================================
# BLOCK 5: STAGE 2 - PURE 1-BIT LSB SHAMIR (High Fidelity)
# ==========================================
class ShamirSecretSharing:
    def __init__(self, prime=257):
        self.prime = prime

    def _mod_inverse(self, a, m):
        if a < 0: a = (a % m + m) % m
        m0, y, x = m, 0, 1
        if m == 1: return 0
        while a > 1:
            q = a // m
            m, a = a % m, m
            y, x = x - q * y, y
        if x < 0: x += m0
        return x

    def split_secret(self, secret, num_shares, threshold):
        while True:
            coeffs = [secret] + [random.randint(0, self.prime - 1) for _ in range(threshold - 1)]
            shares = []
            valid_batch = True
            for i in range(1, num_shares + 1):
                y = sum(c * (i**exp) for exp, c in enumerate(coeffs)) % self.prime
                if y >= 256:
                    valid_batch = False
                    break
                shares.append((i, y))
            if valid_batch: return shares

    def reconstruct_secret(self, shares):
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            xi, yi = int(xi), int(yi)
            num, den = 1, 1
            for j, (xj, _) in enumerate(shares):
                if i == j: continue
                xj = int(xj)
                num = (num * (0 - xj)) % self.prime
                den = (den * (xi - xj)) % self.prime
            lagrange = num * self._mod_inverse(den, self.prime)
            secret = (secret + (yi * lagrange)) % self.prime
        return int(secret)

def embed_shamir_n_stego(cover_img, payload_bits, num_stego_images, threshold=None, exclusion_indices=None):
    rows, cols = cover_img.shape
    num_pixels = rows * cols
    shamir = ShamirSecretSharing(prime=257)

    stego_images = [cover_img.copy() for _ in range(num_stego_images)]

    pad_len = (8 - len(payload_bits) % 8) % 8
    payload_bits_padded = payload_bits + '0' * pad_len
    secrets = [int(payload_bits_padded[i:i+8], 2) for i in range(0, len(payload_bits_padded), 8)]

    if exclusion_indices is not None and len(exclusion_indices) > 0:
        mask = np.ones(num_pixels, dtype=bool)
        mask[exclusion_indices] = False
        available_indices = np.where(mask)[0]
    else:
        available_indices = np.arange(num_pixels)

    np.random.seed(999)
    np.random.shuffle(available_indices)

    total_bits_to_store = len(secrets) * 8
    total_capacity = len(available_indices)

    print(f"    [Stage 2] SSS Mode: Pure 1-bit LSB")
    print(f"    [Stage 2] Capacity Check: Need {total_bits_to_store} bits / Have {total_capacity} pixels")

    if total_bits_to_store > total_capacity:
        print("⚠ CRITICAL FAILURE: Payload exceeds 1-bit capacity!")
        return None, {}

    image_bitstreams = [""] * num_stego_images

    print(f"    [Stage 2] Generating SSS Shares...")
    for secret in secrets:
        shares = shamir.split_secret(secret, num_stego_images, threshold)
        for img_idx, (_, y_share) in enumerate(shares):
            image_bitstreams[img_idx] += format(y_share, '08b')

    print(f"    [Stage 2] Embedding 1-bit Streams...")

    original_lsbs = {}
    for img_idx in range(num_stego_images):
        bitstream = image_bitstreams[img_idx]
        bit_cursor = 0
        img = stego_images[img_idx]

        for idx in available_indices:
            if bit_cursor >= len(bitstream): break

            r, c = idx // cols, idx % cols

            if idx not in original_lsbs:
                original_lsbs[idx] = int(img[r, c] & 1)

            bit_val = int(bitstream[bit_cursor], 2)
            bit_cursor += 1

            img[r, c] = (img[r, c] & 0xFE) | bit_val

    return stego_images, original_lsbs

def extract_shamir_n_stego(stego_images, payload_len_bits, threshold=None, exclusion_indices=None):
    rows, cols = stego_images[0].shape
    num_pixels = rows * cols
    shamir = ShamirSecretSharing(prime=257)

    if exclusion_indices is not None and len(exclusion_indices) > 0:
        mask = np.ones(num_pixels, dtype=bool)
        mask[exclusion_indices] = False
        available_indices = np.where(mask)[0]
    else:
        available_indices = np.arange(num_pixels)

    np.random.seed(999)
    np.random.shuffle(available_indices)

    pad_len = (8 - payload_len_bits % 8) % 8
    total_bits_to_store = payload_len_bits + pad_len

    image_bitstreams = [""] * len(stego_images)

    for img_idx, img in enumerate(stego_images):
        bit_cursor = 0
        extracted = []

        for idx in available_indices:
            if bit_cursor >= total_bits_to_store: break

            r, c = idx // cols, idx % cols

            val = img[r, c] & 1
            extracted.append(str(val))
            bit_cursor += 1

        image_bitstreams[img_idx] = "".join(extracted)

    extracted_bits = ""
    stream_len = len(image_bitstreams[0])

    for i in range(0, stream_len, 8):
        shares_collected = []
        for img_idx in range(len(stego_images)):
            byte_str = image_bitstreams[img_idx][i : i+8]
            if len(byte_str) < 8: break
            y_share = int(byte_str, 2)
            shares_collected.append((img_idx + 1, y_share))

        subset_shares = shares_collected[:threshold] if threshold else shares_collected
        secret = shamir.reconstruct_secret(subset_shares)
        extracted_bits += format(secret, '08b')

    return extracted_bits[:payload_len_bits]

print("Block 5 (Pure 1-bit SSS) loaded successfully.")

# ==========================================
# PAYLOAD PROCESSOR
# ==========================================
class PayloadProcessor:
    def __init__(self):
        self.clean_text = None

    def process_for_embedding(self, secret_text):
        print(f"[Preprocessing] Initial Read Size: {len(secret_text)} chars")
        clean_text = re.sub(r'[^01]', '', secret_text)

        if len(clean_text) > 0:
            print(f"[Preprocessing] Found {len(clean_text)} valid bits. Packing...")
            padding_needed = (8 - len(clean_text) % 8) % 8
            clean_text += '0' * padding_needed
            num_bytes = len(clean_text) // 8
            raw_bytes = int(clean_text, 2).to_bytes(num_bytes, byteorder='big')
            print(f"[Preprocessing] Packed Size: {len(raw_bytes)} bytes (Success)")
        else:
            print(f"[Preprocessing] ⚠ No binary data found. Falling back to UTF-8.")
            raw_bytes = secret_text.encode('utf-8')

        data_bits = ''.join(format(b, '08b') for b in raw_bytes)
        payload_bits = add_length_header(data_bits)
        
        self.clean_text = clean_text 
        self.info = {'orig_len': len(raw_bytes), 'has_rs': False}
        return payload_bits, self.info

    def process_after_extraction(self, extracted_bits):
        data_bits = extract_length_header(extracted_bits)
        if not data_bits:
             return ""
        
        padding = (8 - len(data_bits) % 8) % 8
        padded_bits = data_bits + '0' * padding
        compressed_bytes = bytearray()
        for i in range(0, len(padded_bits), 8):
            compressed_bytes.append(int(padded_bits[i:i+8], 2))
        
        recovered_bytes = bytes(compressed_bytes)
        recovered_bits = ''.join(format(b, '08b') for b in recovered_bytes)
        
        if self.clean_text is not None:
             recovered_bits = recovered_bits[:len(self.clean_text)]
             
        return recovered_bits

    def verify(self, recovered_text):
        if self.clean_text is None:
            return False
        res = (recovered_text == self.clean_text)
        if not res:
            print(f"VERIFY FAILED! clean length={len(self.clean_text)}, recovered length={len(recovered_text)}")
        return res

# ==========================================
# CONFIG
# ==========================================
BASE_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity/Results"
COVER_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity/Cover"
PAYLOAD_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity/payload"

COVER_FILES = [f"image {i}.tiff" for i in range(1, 16)]
PAYLOAD_SIZES = [1] + list(range(10, 101, 10))
PAYLOAD_TEMPLATE = "random-binary_{}Kb.txt"

NUM_STEGO_IMAGES = 4
SHAMIR_THRESHOLD = 3
IMPACT_RATIO =  0.25

embed_results = []
extract_results = []
psnr_results = []
ssim_results = []
bpp_results = []
mse_results = []


# ==========================================
# TIMING FUNCTION
# ==========================================
def run_timing_experiment(cover_img, payload_bits, processor):
    total_bits = len(payload_bits)

    # =========================
    # EMBEDDING
    # =========================
    start_embed = time.perf_counter()

    try:
        stego_inter, indices_ga, avg_table, bits_in_ga = embed_group_average_grey(
            cover_img,
            payload_bits,
            use_impact_analysis=True,
            impact_pool_ratio=IMPACT_RATIO
        )

        bits_remaining = total_bits - bits_in_ga
        payload_part_b = payload_bits[bits_in_ga:]

        if bits_remaining > 0:
            stego_final_list, original_lsbs = embed_shamir_n_stego(
                stego_inter,
                payload_part_b,
                NUM_STEGO_IMAGES,
                SHAMIR_THRESHOLD,
                exclusion_indices=indices_ga
            )
        else:
            stego_final_list = [stego_inter.copy() for _ in range(NUM_STEGO_IMAGES)]
            original_lsbs = {}

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

    embed_time = time.perf_counter() - start_embed

    # =========================
    # EXTRACTION
    # =========================
    start_extract = time.perf_counter()

    try:
        recovered_img, extracted_ga_bits = extract_group_average_grey(
            stego_final_list[0],
            indices_ga,
            avg_table
        )

        if original_lsbs:
            recovered_flat = recovered_img.flatten()
            for idx, orig_lsb in original_lsbs.items():
                recovered_flat[idx] = (recovered_flat[idx] & 0xFE) | orig_lsb
            recovered_img = recovered_flat.reshape(cover_img.shape).astype(np.uint8)

        if len(extracted_ga_bits) < 32:
             print("❌ Extraction error: Not enough bits to read length header.")
             return None
             
        data_length = int(extracted_ga_bits[:32], 2)
        expected_total_bits = data_length + 32
        
        available_ga_bits = len(indices_ga)
        
        if expected_total_bits <= available_ga_bits:
             extracted_bits = extracted_ga_bits[:expected_total_bits]
             extracted_shamir_bits = ""
             payload_part_b = ""
        else:
             extracted_ga_bits = extracted_ga_bits[:available_ga_bits]
             shamir_bits_needed = expected_total_bits - available_ga_bits
             
             extracted_shamir_bits = extract_shamir_n_stego(
                 stego_final_list,
                 payload_len_bits=shamir_bits_needed,
                 threshold=SHAMIR_THRESHOLD,
                 exclusion_indices=indices_ga
             )
             extracted_bits = extracted_ga_bits + extracted_shamir_bits
             payload_part_b = payload_bits[available_ga_bits:]

        print("\\n[BITSTREAM DEBUG]")
        print("Original bits:", total_bits)
        print("Extracted bits:", len(extracted_bits))
        print("GA bits extracted:", len(extracted_ga_bits))
        print("Shamir bits extracted:", len(extracted_shamir_bits))

        bit_mismatch = (extracted_bits != payload_bits)
        if bit_mismatch:
             print("   --> Mismatch found!")
        else:
             print("   --> Bitstream matches perfectly!")

        recovered_text = processor.process_after_extraction(extracted_bits)
        is_correct = processor.verify(recovered_text)

    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return None
    extract_time = time.perf_counter() - start_extract

    metrics = calculate_metrics("Stego Image 1", cover_img, stego_final_list[0], total_bits)

    return embed_time, extract_time, is_correct, metrics


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Timing Experiment...\n")

    for cover_name in COVER_FILES:
        cover_path = os.path.join(COVER_PATH, cover_name)

        cover_img = load_image_safe(cover_path)
        if cover_img is None:
            continue

        print(f"\n📸 Processing {cover_name}")

        for size in PAYLOAD_SIZES:
            payload_path = os.path.join(PAYLOAD_PATH, PAYLOAD_TEMPLATE.format(size))

            if not os.path.exists(payload_path):
                print(f"   ⚠️ Missing payload {size}KB")
                continue

            with open(payload_path, 'r', encoding='utf-8', errors='ignore') as f:
                secret_text = f.read()

            processor = PayloadProcessor()
            payload_bits, _ = processor.process_for_embedding(secret_text)

            result = run_timing_experiment(cover_img, payload_bits, processor)

            if result:
                e_time, x_time, is_correct, metrics = result

                print(f"   ✅ {size}KB -> Embed: {e_time:.4f}s | Extract: {x_time:.4f}s | Correct: {is_correct} | PSNR: {metrics['psnr']:.3f}")

                embed_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "Embed Time": e_time,
                    "Correct": is_correct
                })

                extract_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "Extract Time": x_time,
                    "Correct": is_correct
                })

                mse_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "MSE": metrics['mse']
                })
                
                psnr_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "PSNR": metrics['psnr']
                })
                
                ssim_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "SSIM": metrics['ssim']
                })
                
                bpp_results.append({
                    "Cover Image": cover_name,
                    "Payload Size (KB)": size,
                    "BPP": metrics['bpp']
                })

            else:
                print(f"   ❌ Failed at {size}KB")

    # ==========================================
    # SAVE CSV
    # ==========================================
    df_embed = pd.DataFrame(embed_results)
    df_extract = pd.DataFrame(extract_results)
    df_psnr = pd.DataFrame(psnr_results)
    df_ssim = pd.DataFrame(ssim_results)
    df_bpp = pd.DataFrame(bpp_results)
    df_mse = pd.DataFrame(mse_results)

    embed_path = os.path.join(BASE_PATH, "embed_timing.csv")
    extract_path = os.path.join(BASE_PATH, "extract_timing.csv")
    psnr_path = os.path.join(BASE_PATH, "PSNR.csv")
    ssim_path = os.path.join(BASE_PATH, "SSIM.csv")
    bpp_path = os.path.join(BASE_PATH, "bpp.csv")
    mse_path = os.path.join(BASE_PATH, "MSE.csv")

    try:
        df_embed.to_csv(embed_path, index=False)
        df_extract.to_csv(extract_path, index=False)
        df_psnr.to_csv(psnr_path, index=False)
        df_ssim.to_csv(ssim_path, index=False)
        df_bpp.to_csv(bpp_path, index=False)
        df_mse.to_csv(mse_path, index=False)
        print(f"\n✅ Embed results saved to: {embed_path}")
        print(f"✅ Extract results saved to: {extract_path}")
        print(f"✅ PSNR results saved to: {psnr_path}")
        print(f"✅ SSIM results saved to: {ssim_path}")
        print(f"✅ BPP results saved to: {bpp_path}")
        print(f"✅ MSE results saved to: {mse_path}")
    except Exception as e:
        print(f"Warning could not save CSV: {e}")
