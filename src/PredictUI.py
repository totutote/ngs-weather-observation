import torch
from PIL import Image, ImageTk
from models.weather_classifier import WeatherClassifier
import pyautogui
import pygetwindow as gw
import tkinter as tk
from tkinter import ttk
import threading
import time
from predict import WeatherClassifierPredict

class WeatherClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Classifier")

        self.mainframe = ttk.Frame(root, padding="50 40 50 40")
        self.mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.label = ttk.Label(self.mainframe, text="Predicted class: ")
        self.label.grid(column=1, row=1, sticky=(tk.W, tk.E))

        self.prob_label = ttk.Label(self.mainframe, text="Probability: ")
        self.prob_label.grid(column=1, row=2, sticky=(tk.W, tk.E))

        self.image_label = ttk.Label(self.mainframe)
        self.image_label.grid(column=1, row=3, sticky=(tk.W, tk.E))

        self.start_button = ttk.Button(self.mainframe, text="Start", command=self.start_capture)
        self.start_button.grid(column=1, row=4, sticky=(tk.W, tk.E))

        self.stop_button = ttk.Button(self.mainframe, text="Stop", command=self.stop_capture)
        self.stop_button.grid(column=1, row=5, sticky=(tk.W, tk.E))

        self.interval = 10  # seconds
        self.stop_event = None
        self.capture_thread = None

        # ウィンドウのタイトル
        self.window_title = "PHANTASY STAR ONLINE 2 NEW GENESIS"

        # モデルの初期化
        self.model = WeatherClassifierPredict()

    def periodic_capture_and_predict(self, stop_event):
        while not stop_event.is_set():
            # 指定されたウィンドウを取得
            window = gw.getWindowsWithTitle(self.window_title)[0]
            if window is None:
                raise ValueError(f"Window with title '{self.window_title}' not found")

            # ウィンドウをキャプチャする
            screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))

            # 画像のサイズを変更
            resized_screenshot = screenshot.resize((256, 256))  # ここでサイズを変更

            # 予測ロジック
            result, probability = self.model.predict(resized_screenshot)
            
            # UIの更新
            self.label.config(text=f"Predicted class: {result}")
            self.prob_label.config(text=f"Probability: {probability}")
            
            # スクリーンショットをPIL ImageからTkinter Imageに変換して表示
            img = ImageTk.PhotoImage(resized_screenshot)
            self.image_label.config(image=img)
            self.image_label.image = img  # 参照を保持するために必要

            # 次のキャプチャまで待機
            time.sleep(self.interval)

    def start_periodic_capture(self):
        self.stop_event = threading.Event()
        self.capture_thread = threading.Thread(target=self.periodic_capture_and_predict, args=(self.stop_event,))
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_periodic_capture(self):
        if self.stop_event:
            self.stop_event.set()

    def start_capture(self):
        self.start_periodic_capture()

    def stop_capture(self):
        self.stop_periodic_capture()


if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherClassifierApp(root)
    root.mainloop()
