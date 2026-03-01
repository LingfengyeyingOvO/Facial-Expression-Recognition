

import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv('training_history.csv')

# 绘制损失
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(df['epoch'], df['loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

# 绘制准确率
plt.subplot(1,2,2)
plt.plot(df['epoch'], df['accuracy'], label='Train Acc')
plt.plot(df['epoch'], df['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.show()
# 也可以保存图片
plt.savefig('training_curves.png')