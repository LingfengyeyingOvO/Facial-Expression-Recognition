import numpy as np
import cv2
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Activation

# 路径设置
train_dir = 'data/train'
val_dir = 'data/test'

# 数据增强（可提高泛化能力）
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False  # 表情识别通常不水平翻转，因为左右脸不对称
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# 保存类别索引，供GUI使用
class_indices = train_generator.class_indices
with open('class_names.json', 'w') as f:
    json.dump(class_indices, f)
print("类别映射已保存到 class_names.json")

# 构建模型
emotion_model = Sequential()

# ===== Block 1 =====
emotion_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))

emotion_model.add(Conv2D(64, (3, 3), padding='same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))

emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Dropout(0.25))

# ===== Block 2 =====
emotion_model.add(Conv2D(128, (3, 3), padding='same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))

emotion_model.add(MaxPooling2D((2, 2)))

emotion_model.add(Conv2D(128, (3, 3), padding='same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))

emotion_model.add(MaxPooling2D((2, 2)))
emotion_model.add(Dropout(0.25))

# ===== Classifier =====
emotion_model.add(Flatten())
emotion_model.add(Dense(1024))
emotion_model.add(BatchNormalization())   # Dense层也可以加BN（很加分）
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.5))

emotion_model.add(Dense(train_generator.num_classes, activation='softmax'))

emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# 回调：早停 + 保存最佳模型
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_batchNorm_model.h5', save_best_only=True)
]

history = emotion_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks
)


import pandas as pd

# 提取历史数据
history_dict = history.history
# 转换为 DataFrame
df = pd.DataFrame(history_dict)
# 保存为 CSV 文件（可用 Excel 打开）
df.to_csv('training_batchNorm_history.csv', index=False)
print("训练历史已保存到 training_batchNorm_history.csv")

# 也可以在控制台打印出所有 epoch 的详细数据
print("\n每个 epoch 的详细指标：")
for epoch in range(len(df)):
    print(f"Epoch {epoch+1:2d}: "
          f"loss={df.loc[epoch, 'loss']:.4f}, "
          f"accuracy={df.loc[epoch, 'accuracy']:.4f}, "
          f"val_loss={df.loc[epoch, 'val_loss']:.4f}, "
          f"val_accuracy={df.loc[epoch, 'val_accuracy']:.4f}")



# 保存最终模型（可选）
emotion_model.save('emotion_batchNorm_model.h5')
print("模型已保存到 emotion_batchNorm_model.h5")

import matplotlib.pyplot as plt

# 绘制训练 & 验证损失
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制训练 & 验证准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()  # 在 Colab 中显示图像
# 也可以保存到文件
plt.savefig('training_batchNorm_curves.png')
print("训练曲线已保存为 training_batchNorm_curves.png")