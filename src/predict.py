import torch
from torchvision import transforms
from models.weather_classifier import WeatherClassifier

class WeatherClassifierPredict:
    def __init__(self):
        # モデルのチェックポイントを読み込む
        checkpoint_path = "checkpoints/weather-classifier.ckpt"
        self.model = WeatherClassifier.load_from_checkpoint(checkpoint_path, num_classes=2)
        self.model.eval()  # モデルを評価モードに設定

        # 画像の前処理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # 予測を行う
    def predict(self, image):
        with torch.no_grad():
            image = self.preprocess(image)
            image = image.unsqueeze(0)  # バッチサイズの次元を追加
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item(), probabilities[0][predicted].item()