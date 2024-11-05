import torch
from torchvision import transforms
from PIL import Image
from models.weather_classifier import WeatherClassifier
import pygetwindow as gw
import pyautogui

# モデルのチェックポイントを読み込む
checkpoint_path = "checkpoints/weather-classifier.ckpt"
model = WeatherClassifier.load_from_checkpoint(checkpoint_path, num_classes=10)
model.eval()  # モデルを評価モードに設定

# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def capture_window(window_title, save_path):
    # 指定されたウィンドウを取得
    window = gw.getWindowsWithTitle(window_title)[0]
    if window is None:
        raise ValueError(f"Window with title '{window_title}' not found")

    # ウィンドウをキャプチャする
    window.activate()
    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    screenshot.save(save_path)
    
    # 画像を読み込む
    image = Image.open(save_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # バッチサイズの次元を追加

    return image

# 予測を行う
window_title = "Your Application Window Title"
image_path = "path/to/your/captured_image.jpg"
image = capture_window(window_title, image_path)

# モデルを使って予測を行う
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

predicted = predicted.item()

print(f"Predicted class: {predicted}")