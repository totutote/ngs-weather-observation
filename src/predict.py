import torch
from torchvision import transforms
from PIL import Image, ImageTk
from models.weather_classifier import WeatherClassifier
import pyautogui
import pygetwindow as gw
import tkinter as tk
from tkinter import ttk
import threading
import time

# モデルのチェックポイントを読み込む
checkpoint_path = "checkpoints/weather-classifier.ckpt"
model = WeatherClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
model.eval()  # モデルを評価モードに設定

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def capture_and_predict(save_path):
    # 指定されたウィンドウを取得
    window_title = "PHANTASY STAR ONLINE 2 NEW GENESIS"
    window = gw.getWindowsWithTitle(window_title)[0]
    if window is None:
        raise ValueError(f"Window with title '{window_title}' not found")

    # ウィンドウをキャプチャする
    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    screenshot.save(save_path)
    
    # 画像を読み込む
    image = Image.open(save_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # バッチサイズの次元を追加

    # モデルを使って予測を行う
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probabilities[0][predicted].item()

def periodic_capture_and_predict(interval, save_path, label, prob_label, image_label):
    while True:
        prediction, probability = capture_and_predict(save_path)
        
        # 画像をTkinterのLabelに表示する
        img = Image.open(save_path)
        img = img.resize((224, 224))  # 必要に応じてサイズを調整
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # 予測結果と確率を表示
        label.config(text=f"Predicted class: {'悪天候' if prediction == 0 else '通常'}")
        prob_label.config(text=f"Probability: {probability:.2f}")
        time.sleep(interval)

def start_periodic_capture(interval, save_path, label, prob_label, image_label):
    thread = threading.Thread(target=periodic_capture_and_predict, args=(interval, save_path, label, prob_label, image_label))
    thread.daemon = True
    thread.start()

# GUIの設定
root = tk.Tk()
root.title("Weather Classifier")

mainframe = ttk.Frame(root, padding="60 60 60 60")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label = ttk.Label(mainframe, text="Predicted class: ")
label.grid(column=1, row=1, sticky=(tk.W, tk.E))

prob_label = ttk.Label(mainframe, text="Probability: ")
prob_label.grid(column=1, row=2, sticky=(tk.W, tk.E))

image_label = ttk.Label(mainframe)
image_label.grid(column=1, row=3, sticky=(tk.W, tk.E))

# 定期的にスクリーンショットを撮って予測を行う
interval = 10  # 秒
save_path = "content/predict_img/captured_image.jpg"
start_periodic_capture(interval, save_path, label, prob_label, image_label)

root.mainloop()