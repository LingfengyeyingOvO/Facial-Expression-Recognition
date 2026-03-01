import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ======================
# 参数设置（和你的CNN保持一致）
# ======================
IMG_SIZE = 48
DATA_TRAIN_DIR = "data/train"
DATA_TEST_DIR = "data/test"


# ======================
# 1. 读取数据（适配你的文件夹结构）
# ======================
def load_dataset(data_dir, img_size=48):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    print(f"Loading data from: {data_dir}")
    print(f"Detected classes: {class_names}")

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # 灰度图
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 统一尺寸
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples from {data_dir}")
    return X, y, class_names


# ======================
# 2. 提取 HOG 特征
# ======================
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        hog_features.append(features)
    return np.array(hog_features)


# ======================
# 3. 画归一化 Confusion Matrix（类似你想要的那种热力图）
# ======================
def plot_confusion_matrix(y_true, y_pred, class_names, title, normalize=True):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm / row_sums
        fmt = "{:.2f}"
    else:
        cm_show = cm
        fmt = "{:d}"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_show)
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            plt.text(j, i, fmt.format(cm_show[i, j]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ======================
# 4. 主训练流程：HOG + Softmax（多类Logistic Regression）
# ======================
def main():
    start_time = time.time()
    print("===== HOG + Softmax (Multinomial Logistic Regression) =====")

    # Step 1: 加载数据
    X_train_img, y_train, class_names = load_dataset(DATA_TRAIN_DIR, IMG_SIZE)
    X_test_img, y_test, class_names_test = load_dataset(DATA_TEST_DIR, IMG_SIZE)

    # 安全检查：确保 train/test 类别顺序一致
    if class_names_test != class_names:
        print("\n[Warning] Train/Test class order differs!")
        print("Train classes:", class_names)
        print("Test classes :", class_names_test)
        print("This may break label alignment. Please ensure folder names match.")
        # 不中断，继续跑，但强烈建议你修一致

    # Step 2: HOG 特征
    print("\nExtracting HOG features...")
    X_train_hog = extract_hog_features(X_train_img)
    X_test_hog = extract_hog_features(X_test_img)
    print(f"HOG feature dimension: {X_train_hog.shape[1]}")

    # Step 3 & 4: Standardize + Softmax classifier
    # Softmax = multinomial logistic regression
    # 注意：这里的 “C” 是正则强度的反比（和SVM的C含义方向相同：C越大正则越弱，越容易拟合）
    clf_softmax = Pipeline([
        ("scaler", StandardScaler()),
        ("softmax", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=4000,
            C=1.0
        ))
    ])

    print("\nTraining Softmax classifier (multinomial logistic regression)...")
    clf_softmax.fit(X_train_hog, y_train)

    # Step 5: 预测
    print("\nEvaluating on test set...")
    y_pred = clf_softmax.predict(X_test_hog)

    # Step 6: 评估指标
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest Accuracy (HOG + Softmax): {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix (counts):")
    print(confusion_matrix(y_test, y_pred))

    # 归一化热力图（每行=1，更像你截图那种）
    plot_confusion_matrix(
        y_test, y_pred, class_names,
        title=f"Test Confusion Matrix (HOG + Softmax) Acc={acc:.3f}",
        normalize=True
    )

    end_time = time.time()
    print(f"\nTotal Running Time: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    main()