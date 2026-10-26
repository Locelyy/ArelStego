import os
import time
import numpy as np
import pandas as pd
import sys
import cv2

# Import exact methodology from GAPS.py
from GAPS import load_image_safe, embed_group_average_grey, extract_group_average_grey, calculate_metrics, PayloadProcessor

BASE_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity"
COVER_PATH = os.path.join(BASE_PATH, "Cover")
PAYLOAD_PATH = os.path.join(BASE_PATH, "payload")
STEGO_OUTPUT_PATH = os.path.join(BASE_PATH, "SingleOutputStego")

# Create a folder to store the single output stego images
os.makedirs(STEGO_OUTPUT_PATH, exist_ok=True)

COVER_FILES = [f"image {i}.tiff" for i in range(1, 16)]
IMPACT_RATIO = 0.1

def main():
    print("Initializing Single Output Experiment (1KB Payload)...")
    
    processor = PayloadProcessor()
    payload_path = os.path.join(PAYLOAD_PATH, "random-binary_1Kb.txt")
    
    if not os.path.exists(payload_path):
        print(f"Error: Could not find payload at {payload_path}")
        return
        
    with open(payload_path, 'r', encoding='utf-8', errors='ignore') as f:
        secret_text = f.read()
        
    payload_bits, _ = processor.process_for_embedding(secret_text)
    total_bits = len(payload_bits)
    print(f"Loaded 1KB payload ({total_bits} bits including header).")

    results = []

    for cover_name in COVER_FILES:
        cover_path = os.path.join(COVER_PATH, cover_name)
        cover_img = load_image_safe(cover_path)
        if cover_img is None: 
            print(f"Warning: Missing {cover_name}")
            continue
            
        print(f"\nProcessing {cover_name}...")
        
        # ==========================================
        # EMBEDDING (GA Stage 1 ONLY)
        # ==========================================
        start_embed = time.perf_counter()
        
        # Suppress prints from GAPS.py to keep terminal clean
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        
        try:
            stego_img, indices_ga, avg_table, bits_in_ga = embed_group_average_grey(
                cover_img,
                payload_bits,
                use_impact_analysis=True,
                impact_pool_ratio=IMPACT_RATIO
            )
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        embed_time = time.perf_counter() - start_embed
        
        if bits_in_ga < total_bits:
            print(f"   WARNING: Payload exceeded GA capacity! Bits embedded: {bits_in_ga}/{total_bits}")
        
        # Save the single output image
        out_img_name = f"stego_{cover_name}"
        out_img_path = os.path.join(STEGO_OUTPUT_PATH, out_img_name)
        cv2.imwrite(out_img_path, stego_img)
        
        # ==========================================
        # EXTRACTION (From the Single Output)
        # ==========================================
        start_extract = time.perf_counter()
        
        recovered_img, extracted_ga_bits = extract_group_average_grey(
            stego_img,
            indices_ga,
            avg_table
        )
        
        # Parse data length from the header
        if len(extracted_ga_bits) >= 32:
            data_length = int(extracted_ga_bits[:32], 2)
            expected_total_bits = data_length + 32
            extracted_bits = extracted_ga_bits[:expected_total_bits]
        else:
            extracted_bits = extracted_ga_bits
            
        # Validate exact match
        recovered_text = processor.process_after_extraction(extracted_bits)
        is_correct = processor.verify(recovered_text)
        
        extract_time = time.perf_counter() - start_extract
        
        # Calculate Metrics
        metrics = calculate_metrics(cover_name, cover_img, stego_img, total_bits)
        
        results.append({
            "Cover Image": cover_name,
            "PSNR (dB)": metrics['psnr'],
            "SSIM": metrics['ssim'],
            "BPP": metrics['bpp'],
            "Embed Time (s)": embed_time,
            "Extract Time (s)": extract_time,
            "Extraction Success": is_correct
        })
        
        print(f"   Success: {is_correct} | PSNR: {metrics['psnr']:.2f} dB | Embed Time: {embed_time:.3f}s")
        
    # ==========================================
    # SAVE METRICS
    # ==========================================
    df = pd.DataFrame(results)
    csv_path = os.path.join(BASE_PATH, "single_output_1kb_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\nExperiment complete! Saved results to {csv_path}")
    print(f"Saved {len(results)} stego images to the folder: {STEGO_OUTPUT_PATH}")
    
    print("\n--- Average Metrics ---")
    print(df[["PSNR (dB)", "SSIM", "Embed Time (s)", "Extract Time (s)"]].mean().to_string())

if __name__ == "__main__":
    main()
