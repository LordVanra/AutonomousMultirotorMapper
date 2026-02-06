import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Data
data = np.array([
    [88, 96, 82, 41],
    [88, 95, 80, 45]
])

confidence_levels = ['0.1', '0.2', '0.3', '0.4']
resolutions = ['720p', '1080p']

# Custom colors (RGB → 0–1)
low_color = (188/255, 207/255, 230/255)
high_color = (24/255, 56/255, 95/255)

cmap = LinearSegmentedColormap.from_list("custom_blue", [low_color, high_color])
norm = Normalize(vmin=data.min(), vmax=data.max())

# Plot
plt.figure()
img = plt.imshow(data, cmap=cmap, norm=norm)

# Axis labels and ticks
plt.xticks(np.arange(len(confidence_levels)), confidence_levels, fontsize=24)
plt.yticks(np.arange(len(resolutions)), resolutions, fontsize=24)

plt.xlabel("Confidence Threshold", fontsize=24)
plt.ylabel("Image Resolution", fontsize=24)
plt.title("Detection Accuracy Heatmap", fontsize=24)

# Colorbar
cbar = plt.colorbar(img)
cbar.set_label("Accuracy (%)", fontsize=24)
cbar.ax.tick_params(labelsize=24)

# Cell annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, f"{data[i, j]}%", 
                 ha='center', va='center', fontsize=24)

plt.tight_layout()
plt.show()
