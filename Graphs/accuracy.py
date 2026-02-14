import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("path.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Red corridor
red_mask = (
    (hsv[:,:,1] > 40) &
    ((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165))
).astype(np.uint8)

# Blue path
blue_mask = (
    (hsv[:,:,0] > 90) &
    (hsv[:,:,0] < 140) &
    (hsv[:,:,1] > 40)
).astype(np.uint8)

# --- Extract blue path points ---
ys, xs = np.where(blue_mask == 1)
coords = np.column_stack([xs, ys])

order = np.lexsort((-coords[:,0], -coords[:,1]))
coords = coords[order]

accuracies = []

for x, y in coords:
    band = red_mask[max(0,y-3):min(img.shape[0],y+4), :]
    xs_red = np.where(band == 1)[1]

    if len(xs_red) < 30:
        continue

    left, right = xs_red.min(), xs_red.max()
    center = (left + right) / 2
    half_width = (right - left) / 2

    # Linear accuracy
    raw_accuracy = max(0, 1 - abs(x - center) / half_width)
    # Apply 1/8 power scaling
    scaled_accuracy = raw_accuracy ** (1/8)
    accuracies.append(scaled_accuracy * 100)

accuracies = np.array(accuracies)

cumulative_accuracy = np.cumsum(accuracies) / (np.arange(len(accuracies)) + 1)

df = pd.DataFrame({
    "time_index": np.arange(len(accuracies)),
    "instant_accuracy_percent": accuracies,
    "cumulative_accuracy_percent": cumulative_accuracy
})

df.to_csv("accuracy_over_time.csv", index=False)

plt.figure(figsize=(10,6))
plt.plot(cumulative_accuracy, linewidth=3)
plt.xlabel("Time", fontsize=40)
plt.ylabel("Accuracy (%)", fontsize=40)
plt.title("Accuracy vs Time", fontsize=45)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
