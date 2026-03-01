import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 归一化后的 RBF Confusion Matrix
# -----------------------------
cm = np.array([
    [0.573, 0.014, 0.104, 0.050, 0.094, 0.144, 0.021],
    [0.207, 0.631, 0.027, 0.018, 0.036, 0.072, 0.009],
    [0.117, 0.002, 0.501, 0.029, 0.101, 0.169, 0.081],
    [0.023, 0.000, 0.025, 0.851, 0.051, 0.031, 0.019],
    [0.077, 0.002, 0.057, 0.071, 0.617, 0.161, 0.014],
    [0.114, 0.003, 0.115, 0.055, 0.159, 0.540, 0.014],
    [0.036, 0.001, 0.076, 0.043, 0.020, 0.025, 0.798]
])

class_names = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.colorbar()

plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
plt.yticks(range(len(class_names)), class_names)

# 在格子中显示数值
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 ha="center", va="center")

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Normalized Confusion Matrix (CNN Embedding + RBF SVM)\nAccuracy ≈ 0.66")

plt.tight_layout()
plt.show()