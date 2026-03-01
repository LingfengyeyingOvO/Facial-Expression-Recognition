import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

# ---------- 路径设置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
EMOJI_DIR = os.path.join(BASE_DIR, "emojis")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")

# ---------- 加载类别映射 ----------
if not os.path.exists(CLASS_NAMES_PATH):
    print("错误：找不到 class_names.json，请先运行 train.py 生成该文件。")
    exit(1)

with open(CLASS_NAMES_PATH, 'r') as f:
    class_indices = json.load(f)           # 例如 {"angry":0, "disgust":1, ...}
# 反转得到 {0: "angry", 1: "disgust", ...}
emotion_dict = {v: k for k, v in class_indices.items()}
num_classes = len(emotion_dict)
print("情绪索引映射：", emotion_dict)

# ---------- 定义标签到表情图片文件名的映射（可自定义）----------
# 如果图片文件名与标签名不完全一致，请在此处修改
label_to_emoji_file = {
    'angry': 'angry.png',
    'disgust': 'disgusted.png',      # 文件夹名可能为 disgust，图片名为 disgusted.png
    'fear': 'fearful.png',
    'happy': 'happy.png',
    'neutral': 'neutral.png',
    'sad': 'sad.png',
    'surprise': 'surprised.png'      # 文件夹名可能为 surprise，图片名为 surprised.png
}

# ---------- 预加载表情图片（一次性读入内存）----------
emoji_images = {}
missing_emojis = []
for idx, label in emotion_dict.items():
    filename = label_to_emoji_file.get(label, label + '.png')  # 默认 label.png
    path = os.path.join(EMOJI_DIR, filename)
    if not os.path.exists(path):
        missing_emojis.append(filename)
        continue
    # 使用 PIL 打开并转换为 PhotoImage
    pil_img = Image.open(path).resize((200, 200), Image.Resampling.LANCZOS)  # 统一大小
    emoji_images[idx] = ImageTk.PhotoImage(pil_img)

if missing_emojis:
    print("警告：以下表情图片缺失：", missing_emojis)
    print("请确保 emojis 文件夹中包含对应的图片文件。")

# ---------- 加载模型 ----------
if not os.path.exists(MODEL_PATH):
    print(f"错误：模型文件 {MODEL_PATH} 不存在，请先训练。")
    exit(1)

emotion_model = load_model(MODEL_PATH)

# ---------- 人脸检测器 ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- 全局变量 ----------
current_emotion_idx = 0          # 当前预测的情绪索引
frame_counter = 0                # 用于跳帧
PREDICT_EVERY = 5                # 每5帧预测一次

# ---------- 摄像头 ----------
cap = cv2.VideoCapture(0)

# ---------- GUI 初始化 ----------
root = tk.Tk()
root.title("Photo To Emoji")
root.geometry("1400x900+100+10")
root['bg'] = 'black'

# 标题
heading2 = Label(root, text="Photo to Emoji", pady=20,
                 font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
heading2.pack()

# 视频显示区域
lmain = Label(master=root, padx=50, bd=10)
lmain.place(x=50, y=250)

# 表情图片显示区域
lmain2 = Label(master=root, bd=10)
lmain2.place(x=900, y=350)

# 情绪文字显示区域
lmain3 = Label(master=root, bd=10, fg="#CDCDCD", bg='black', font=('arial', 45, 'bold'))
lmain3.place(x=960, y=250)

# 退出按钮
Button(root, text='Quit', fg="red", command=root.quit,
       font=('arial', 25, 'bold')).pack(side=tk.BOTTOM)

# ---------- 视频更新函数 ----------
def update_video():
    global current_emotion_idx, frame_counter

    ret, frame = cap.read()
    if not ret:
        lmain.after(10, update_video)
        return

    frame = cv2.resize(frame, (600, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 每 PREDICT_EVERY 帧做一次预测，其他帧沿用上次结果
    if frame_counter % PREDICT_EVERY == 0:
        for (x, y, w, h) in faces:
            # 提取人脸 ROI
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            roi_norm = np.expand_dims(np.expand_dims(roi_resized, -1), 0).astype('float32') / 255.0

            preds = emotion_model.predict(roi_norm, verbose=0)
            current_emotion_idx = int(np.argmax(preds))
            break  # 只取检测到的第一张脸

    frame_counter += 1

    # 绘制人脸框（防止坐标出界）
    for (x, y, w, h) in faces:
        y1 = max(0, y - 50)
        y2 = min(frame.shape[0], y + h + 10)
        cv2.rectangle(frame, (x, y1), (x + w, y2), (255, 0, 0), 2)

    # 更新摄像头画面
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    # 更新表情图片（如果已预加载）
    if current_emotion_idx in emoji_images:
        lmain2.configure(image=emoji_images[current_emotion_idx])
    else:
        lmain2.configure(image='', text='No emoji', fg='red')

    # 更新情绪文字
    lmain3.configure(text=emotion_dict.get(current_emotion_idx, 'Unknown'))

    lmain.after(10, update_video)

# ---------- 关闭窗口时的清理 ----------
def on_close():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

# 启动视频更新
update_video()
root.mainloop()