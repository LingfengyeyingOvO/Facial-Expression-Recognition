import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt


# -----------------------------
# Plot: Normalized Confusion Matrix (heatmap)
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", normalize=True):
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


# -----------------------------
# Plot: PCA scatter (PC1 vs PC2)
# -----------------------------
def plot_pca_scatter(X, y, class_names, title="PCA of CNN Embeddings (PC1 vs PC2)", max_points=6000):
    # 可选：下采样，不然 28709 点会比较慢
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], size=max_points, replace=False)
        Xp = X[idx]
        yp = y[idx]
    else:
        Xp = X
        yp = y

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xp)

    plt.figure(figsize=(8, 6))
    for k, name in enumerate(class_names):
        mask = (yp == k)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6, label=name)

    var_ratio = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
    plt.title(title)
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    return pca


# -----------------------------
# Quick check: "linear-ish separability" on PCA(2D)
# This is NOT a proof. It is a diagnostic.
# -----------------------------
def pca_linear_separability_check(X_train, y_train, X_val, y_val):
    pca = PCA(n_components=2, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_val = pca.transform(X_val)

    # 在2D PCA空间做一个简单线性分类器（LogReg）
    # 如果2D就能分得不错，说明主方向上已经有较强可分性；
    # 如果2D效果很差，但高维Linear SVM还不错，说明可分性在更高维子空间。
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, multi_class="auto"))
    ])
    clf.fit(Z_train, y_train)
    pred = clf.predict(Z_val)
    acc = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, average="macro")

    print("\n==== PCA(2D) + Linear Logistic Regression (diagnostic) ====")
    print(f"Val Acc (PCA2D): {acc:.4f}")
    print(f"Macro-F1 (PCA2D): {macro_f1:.4f}")
    print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)

    return pca, acc, macro_f1


def main():
    train_dir = "data/train"
    val_dir = "data/test"
    batch_size = 64

    # 1) 预处理必须和CNN训练一致
    datagen = ImageDataGenerator(rescale=1. / 255)

    # 2) 抽特征时必须 shuffle=False（否则特征和标签会错位）
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    # 类别名：确保和 generator 的 class_indices 一致
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("\nClass order (index -> name):", class_names)

    # 3) 加载验证集最优权重模型
    cnn = load_model("best_model.h5")

    # build / call once so cnn has defined inputs
    _ = cnn(tf.zeros((1, 48, 48, 1)))

    # 4) 取 Dense(1024) 作为 embedding（你之前用 -3，这里保持一致）
    feature_model = Model(inputs=cnn.inputs, outputs=cnn.layers[-3].output)

    # 5) 抽特征
    X_train = feature_model.predict(train_gen, verbose=1)
    y_train = train_gen.classes  # 0~6

    X_val = feature_model.predict(val_gen, verbose=1)
    y_val = val_gen.classes

    # -----------------------------
    # A) CNN feature + Linear SVM
    # -----------------------------
    clf_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0))
    ])
    clf_linear.fit(X_train, y_train)
    pred_linear = clf_linear.predict(X_val)

    print("\n==== CNN feature + Linear SVM ====")
    print("Val Acc:", accuracy_score(y_val, pred_linear))
    print("Macro-F1:", f1_score(y_val, pred_linear, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_val, pred_linear))
    print(classification_report(y_val, pred_linear, target_names=class_names))

    plot_confusion_matrix(
        y_val, pred_linear, class_names,
        title=f"Val Confusion Matrix (CNN Embedding + Linear SVM) Acc={accuracy_score(y_val, pred_linear):.3f}",
        normalize=True
    )

    # -----------------------------
    # B) CNN feature + RBF SVM (slow)
    # -----------------------------
    clf_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])
    clf_rbf.fit(X_train, y_train)
    pred_rbf = clf_rbf.predict(X_val)

    print("\n==== CNN feature + RBF SVM ====")
    print("Val Acc:", accuracy_score(y_val, pred_rbf))
    print("Macro-F1:", f1_score(y_val, pred_rbf, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_val, pred_rbf))
    print(classification_report(y_val, pred_rbf, target_names=class_names))

    plot_confusion_matrix(
        y_val, pred_rbf, class_names,
        title=f"Val Confusion Matrix (CNN Embedding + RBF SVM) Acc={accuracy_score(y_val, pred_rbf):.3f}",
        normalize=True
    )

    # -----------------------------
    # C) PCA embedding analysis
    # -----------------------------
    # 1) PCA scatter plot (train embedding)
    plot_pca_scatter(
        X_train, y_train, class_names,
        title="PCA of CNN Embeddings (Train set, PC1 vs PC2)",
        max_points=6000
    )

    # 2) Diagnostic: PCA(2D) linear separability check
    # (如果2D线性就很差，不代表高维不可分，只是说明主2维不够)
    pca_linear_separability_check(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()