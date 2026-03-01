import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

# ======================
# 参数设置（和你的CNN保持一致）
# ======================
IMG_SIZE = 48  # 你的数据是48x48
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

            # 读取灰度图（和FER标准一致）
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 统一尺寸（控制变量，和CNN一致）
            img = cv2.resize(img, (img_size, img_size))

            X.append(img)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples from {data_dir}")
    return X, y, class_names


# ======================
# 2. 提取 HOG 特征（核心替代CNN部分）
# ======================
def extract_hog_features(images):
    hog_features = []

    for img in images:
        features = hog(
            img,
            orientations=9,          # 方向bin数（经典设置）
            pixels_per_cell=(8, 8),  # 每个cell大小
            cells_per_block=(2, 2),  # block归一化
            block_norm='L2-Hys'
        )
        hog_features.append(features)

    return np.array(hog_features)


# ======================
# 3. 主训练流程（完全替代CNN训练）
# ======================
def main():
    start_time = time.time()

    print("===== HOG + SVM Emotion Recognition Training =====")

    # Step 1: 加载数据（同一数据管线）
    X_train_img, y_train, class_names = load_dataset(DATA_TRAIN_DIR, IMG_SIZE)
    X_test_img, y_test, _ = load_dataset(DATA_TEST_DIR, IMG_SIZE)

    # Step 2: 提取HOG特征（替代CNN特征提取）
    print("\nExtracting HOG features...")
    X_train_hog = extract_hog_features(X_train_img)
    X_test_hog = extract_hog_features(X_test_img)

    print(f"HOG feature dimension: {X_train_hog.shape[1]}")

    # Step 3: 特征标准化（传统ML必须做）
    scaler = StandardScaler()
    X_train_hog = scaler.fit_transform(X_train_hog)
    X_test_hog = scaler.transform(X_test_hog)

    # Step 4: 训练分类器（SVM，比Softmax更适合HOG）
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train_hog, y_train)

    # Step 5: 预测
    print("\nEvaluating on test set...")
    y_pred = svm.predict(X_test_hog)

    # Step 6: 评估指标（用于课程报告）
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy (HOG + SVM): {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    end_time = time.time()
    print(f"\nTotal Training Time: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    main()