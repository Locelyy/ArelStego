import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_PATH = "c:/Users/Arel/Documents/Penelitian/Code/AntiGravity/Results"
CSV_PATH  = os.path.join(BASE_PATH, "SSIM.csv")
OUT_DIR   = os.path.join(BASE_PATH, "graphs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

images = sorted(df["Cover Image"].unique(),
                key=lambda x: int(x.split()[1].split(".")[0]))

cmap   = plt.get_cmap("tab20")
colors = {img: cmap(i % 20) for i, img in enumerate(images)}

fig, ax = plt.subplots(figsize=(13, 6))

for img in images:
    sub = df[df["Cover Image"] == img].sort_values("Payload Size (KB)")
    label = img.replace(".tiff", "").replace("image", "Image")
    ax.plot(sub["Payload Size (KB)"], sub["SSIM"],
            marker="o", markersize=4, linewidth=1.5,
            color=colors[img], label=label)

ax.set_xlabel("Payload Size (KB)", fontsize=12)
ax.set_ylabel("SSIM", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
          fontsize=9, framealpha=0.8, title="Cover Image")

plt.tight_layout()

all_path = os.path.join(OUT_DIR, "SSIM_all_images.png")
fig.savefig(all_path, dpi=150, bbox_inches="tight")
print(f"Saved: {all_path}")
plt.show()

print("Done.")
