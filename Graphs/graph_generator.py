import matplotlib.pyplot as plt
import numpy as np

# Categories
methods = [
    "Color Bounding\n(38/40)",
    "Pretrained Classifier\n(14/40)",
    "YOLO\n(40/40)"
]

# Data
accuracy = [95, 35, 100]      # %
direction = [18, 18, 93]     # %
latency_ms = [7, 142, 100]    # ms

# X positions
x = np.arange(len(methods))
width = 0.25

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

# Bars
bars_accuracy = ax1.bar(
    x - width,
    accuracy,
    width,
    color="orange",
    label="Accuracy (%)"
)

bars_direction = ax1.bar(
    x,
    direction,
    width,
    color="green",
    label="Direction (%)"
)

bars_latency = ax2.bar(
    x + width,
    latency_ms,
    width,
    color="blue",
    label="Time (ms)"
)

# Titles and labels with font size 24
ax1.set_title("Road Detection Accuracy vs Time", fontsize=24)
ax1.set_xlabel("Method", fontsize=24)
ax1.set_ylabel("Road Detection Accuracy (%)", fontsize=24)
ax2.set_ylabel("Time (ms)", fontsize=24)

# X ticks with font size 24
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=24)

# Y ticks with font size 24
ax1.tick_params(axis='y', labelsize=24)
ax2.tick_params(axis='y', labelsize=24)

# Axis limits
ax1.set_ylim(0, 110)
ax2.set_ylim(0, 160)

# Grid (left axis only)
ax1.grid(axis="y", linestyle="--", alpha=0.6)

# Combined legend with font size 24
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=24)

plt.tight_layout()
plt.show()