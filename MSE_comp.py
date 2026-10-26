import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data
images = ['Image 12', 'Image 13', 'Image 14']
x = np.arange(len(images))

# Values extracted from the chart
reyhealth     = [0.002, 0.002, 0.002]
ahmad         = [0.005, 0.003, 0.002]
hameed        = [0.003, 0.004, 0.003]
hussain       = [0.020, 0.010, 0.020]
karakus       = [0.010, 0.010, 0.010]
arelstego     = reyhealth.copy()  # Same values as REYHealth

fig, ax = plt.subplots(figsize=(8, 5))

# Plot each line
ax.plot(x, reyhealth,  color='#1f77b4', marker='o', linewidth=1.5, label='REYHealth')
ax.plot(x, ahmad,      color='#d62728', marker='o', linewidth=1.5, label='Ahmad et al.')
ax.plot(x, hameed,     color='#ff7f0e', marker='o', linewidth=1.5, label='Hameed et al.')
ax.plot(x, hussain,    color='#9467bd', marker='o', linewidth=1.5, label='Hussain & Khodher')
ax.plot(x, karakus,    color='#2ca02c', marker='o', linewidth=1.5, label='Karakus & Avci')
ax.plot(x, arelstego,  color='#8c564b', marker='s', linewidth=1.5, linestyle='--', label='ArelStego')

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(images)
ax.set_xlabel('Image', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_ylim(0, 0.022)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.002))

# Grid
ax.grid(True, linestyle='-', linewidth=0.5, color='#cccccc', alpha=0.8)
ax.set_axisbelow(True)

# Legend outside right
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True,
          fontsize=9, borderaxespad=0)

plt.tight_layout()
plt.savefig(r"C:\Users\Arel\Documents\Penelitian\Code\AntiGravity\Results\Figures\mse_chart.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved.")