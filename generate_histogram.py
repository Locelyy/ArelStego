import cv2
import os
import sys
import io
import numpy as np
from matplotlib import pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import GAPS

# --- CONFIGURATION ---
COVER_IMAGE_NAME = "image 1.tiff"
# For histogram analysis, a larger payload (e.g., 50KB or 100KB) is recommended 
# so you can actually visually observe the changes in the pixel distribution.
PAYLOAD_SIZE = 100 
NUM_STEGO = 4 # Default from GAPS.py

def main():
    cover_path = os.path.join(GAPS.COVER_PATH, COVER_IMAGE_NAME)
    payload_path = os.path.join(GAPS.PAYLOAD_PATH, GAPS.PAYLOAD_TEMPLATE.format(PAYLOAD_SIZE))

    print(f"Loading cover image: {cover_path}")
    cover_img = GAPS.load_image_safe(cover_path)
    if cover_img is None:
        return

    print(f"Loading payload: {payload_path}")
    if not os.path.exists(payload_path):
        print("Payload not found. Please check your payload size.")
        return
        
    with open(payload_path, 'r', encoding='utf-8', errors='ignore') as f:
        secret_text = f.read()

    processor = GAPS.PayloadProcessor()
    payload_bits, _ = processor.process_for_embedding(secret_text)
    
    total_bits = len(payload_bits)
    print(f"Total payload bits to embed: {total_bits}")

    print("\n--- Starting Embedding Process ---")
    # GA Embedding Stage
    stego_inter, indices_ga, avg_table, bits_in_ga = GAPS.embed_group_average_grey(
        cover_img,
        payload_bits,
        use_impact_analysis=True,
        impact_pool_ratio=GAPS.IMPACT_RATIO
    )
    
    bits_remaining = total_bits - bits_in_ga
    payload_part_b = payload_bits[bits_in_ga:]
    
    if bits_remaining > 0:
        print(f"\nPayload overflow! {bits_remaining} bits moving to SSS (Shamir).")
        stego_final_list, _ = GAPS.embed_shamir_n_stego(
            stego_inter,
            payload_part_b,
            NUM_STEGO,
            GAPS.SHAMIR_THRESHOLD,
            exclusion_indices=indices_ga
        )
    else:
        print("\nPayload completely fit in the GA stage.")
        # Duplicate GA stego to match NUM_STEGO for consistency
        stego_final_list = [stego_inter.copy() for _ in range(NUM_STEGO)]

    # Save the resulting images for further use if needed
    base_name = os.path.splitext(COVER_IMAGE_NAME)[0]
    
    cv2.imwrite(f"cover_{base_name}.tiff", cover_img)
    print(f"\nSaved 'cover_{base_name}.tiff' to the current directory.")
    
    for i, stego_img in enumerate(stego_final_list):
        stego_filename = f"{base_name}_N{NUM_STEGO}_{i+1:03d}.tiff"
        cv2.imwrite(stego_filename, stego_img)
        print(f"Saved '{stego_filename}' to the current directory.")
    
    print("\n--- Generating Histogram Comparison Plot ---")
    plt.figure(figsize=(14, 6))
    
    # Plot Cover Histogram
    plt.subplot(1, 2, 1)
    plt.hist(cover_img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(f'Cover Image Histogram ({COVER_IMAGE_NAME})')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Plot Stego Histogram (Using the first stego image)
    plt.subplot(1, 2, 2)
    plt.hist(stego_final_list[0].ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.title(f'Stego Image 1 Histogram ({PAYLOAD_SIZE}KB Payload)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    plt.tight_layout()
    plot_filename = f'histogram_comparison_{base_name}.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved histogram plot to '{plot_filename}'.")

if __name__ == "__main__":
    main()
