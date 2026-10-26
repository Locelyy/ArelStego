import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import from GAPS.py directly to ensure exact methodology
from GAPS import load_image_safe, calculate_metrics, PayloadProcessor, embed_group_average_grey

BASE_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity"
COVER_PATH = os.path.join(BASE_PATH, "Cover")
PAYLOAD_PATH = os.path.join(BASE_PATH, "payload")
PAYLOAD_TEMPLATE = "random-binary_{}Kb.txt"
COVER_FILES = [f"image {i}.tiff" for i in range(1, 16)]

PAYLOAD_SIZES = [1] + list(range(10, 101, 10))
IMPACT_RATIOS = [round(x * 0.1, 1) for x in range(1, 11)]

def main():
    print("Initializing Exact Methodology Impact Ratio vs Payload Sweep...")
    
    # 1. Pre-load payloads
    print("Loading payloads (1KB - 100KB)...")
    payload_data = {}
    processor = PayloadProcessor()
    
    import sys
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    
    for size in PAYLOAD_SIZES:
        payload_path = os.path.join(PAYLOAD_PATH, PAYLOAD_TEMPLATE.format(size))
        if os.path.exists(payload_path):
            with open(payload_path, 'r', encoding='utf-8', errors='ignore') as f:
                secret_text = f.read()
            payload_bits, _ = processor.process_for_embedding(secret_text)
            payload_data[size] = payload_bits
            
    sys.stdout.close()
    sys.stdout = original_stdout
    print(f"Loaded {len(payload_data)} payload sizes.")

    results = []

    # 2. Main Loop
    print("\nWARNING: This uses the exact original GAPS.py methodology.")
    print("Because it re-calculates the edge/variance map for every single payload and ratio,")
    print("it will process 1,650 combinations. This may take around 1 hour to complete.")
    print("Press CTRL+C if you wish to cancel.\n")

    for cover_name in COVER_FILES:
        cover_path = os.path.join(COVER_PATH, cover_name)
        cover_img = load_image_safe(cover_path)
        if cover_img is None: 
            print(f"Warning: Missing {cover_name}")
            continue
        
        print(f"📸 Processing {cover_name}...")
        
        for ratio in IMPACT_RATIOS:
            print(f"   -> Testing Impact Ratio: {ratio}")
            for size in PAYLOAD_SIZES:
                if size not in payload_data: continue
                payload_bits = payload_data[size]
                
                # Suppress the spammy prints from GAPS.py for cleaner terminal output
                sys.stdout = open(os.devnull, 'w', encoding='utf-8')
                
                try:
                    # Uses the EXACT original function from your paper
                    stego_img, indices_ga, avg_table, bits_embedded = embed_group_average_grey(
                        cover_img,
                        payload_bits,
                        use_impact_analysis=True,
                        impact_pool_ratio=ratio
                    )
                except Exception as e:
                    sys.stdout.close()
                    sys.stdout = original_stdout
                    print(f"      Error with payload {size}KB at ratio {ratio}: {e}")
                    continue
                finally:
                    # Always restore stdout
                    if not sys.stdout.closed and sys.stdout != original_stdout:
                        sys.stdout.close()
                    sys.stdout = original_stdout
                
                metrics = calculate_metrics(f"Ratio {ratio}", cover_img, stego_img, bits_embedded)
                results.append({
                    "Image": cover_name,
                    "Ratio": ratio,
                    "Payload Size (KB)": size,
                    "PSNR (dB)": metrics['psnr'],
                    "SSIM": metrics['ssim'],
                    "Capacity Used (Bits)": bits_embedded
                })
                
    df = pd.DataFrame(results)

    # 3. Aggregate Data
    print("\nCalculating Averages...")
    
    grouped_df = df.groupby(["Ratio", "Payload Size (KB)"])[["PSNR (dB)", "SSIM", "Capacity Used (Bits)"]].mean().reset_index()
    csv_path = os.path.join(BASE_PATH, "impact_ratio_payload_ablation.csv")
    grouped_df.to_csv(csv_path, index=False)
    print(f"Full CSV data saved to {csv_path}")

    overall_df = df.groupby("Ratio")[["PSNR (dB)", "SSIM"]].mean().reset_index()

    # 4. Generate Graph
    print("Generating Graph...")
    plt.figure(figsize=(10, 6), facecolor='white')
    
    plt.plot(overall_df["Ratio"], overall_df["PSNR (dB)"], marker='o', linewidth=2.5, color='#1f77b4', label='Average PSNR (All Payloads)')
    
    if 1 in payload_data:
        p1 = grouped_df[grouped_df["Payload Size (KB)"] == 1]
        plt.plot(p1["Ratio"], p1["PSNR (dB)"], marker='s', linestyle='--', alpha=0.7, color='#2ca02c', label='1KB Payload PSNR')
    
    if 100 in payload_data:
        p100 = grouped_df[grouped_df["Payload Size (KB)"] == 100]
        plt.plot(p100["Ratio"], p100["PSNR (dB)"], marker='^', linestyle='--', alpha=0.7, color='#d62728', label='100KB Payload PSNR')

    plt.title("Impact Ratio vs PSNR (Averaged across 15 Images)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Impact Ratio (0.1 = Top 10% Safest Pixels)", fontsize=12)
    plt.ylabel("Peak Signal-to-Noise Ratio (dB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    graph_path = os.path.join(BASE_PATH, "impact_ratio_graph.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved to {graph_path}")

if __name__ == "__main__":
    main()
