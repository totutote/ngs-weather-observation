import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from models.weather_classifier import WeatherClassifier

# データの前処理
weights = ResNet18_Weights.IMAGENET1K_V1
mean = weights.transforms().mean
std = weights.transforms().std

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)

val_test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)

data_dir = "./content/dataimagegenerator_input/"
full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"))

# データセットをランダムにシャッフルして分割
full_size = len(full_dataset)
val_size = int(0.2 * full_size)
test_size = int(0.2 * full_size)
train_size = full_size - test_size - val_size

train_val_dataset, test_dataset = random_split(
    full_dataset, [train_size + val_size, test_size]
)
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# データセットに変換を適用
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# データローダーの作成
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True
)

# モデルの初期化
num_classes = len(full_dataset.classes)
model = WeatherClassifier(num_classes)

# トレーナーの設定
if torch.backends.mps.is_available():
    accelerator = "mps"
elif torch.cuda.is_available():
    accelerator = "gpu"
else:
    accelerator = "cpu"

# CSVLoggerの設定
csv_logger = CSVLogger("logs", name="weather_classification")


# ModelCheckpointの設定
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="weather-classifier",
    mode="min",
)

trainer = Trainer(max_epochs=5, accelerator=accelerator, devices=1, logger=csv_logger, callbacks=[checkpoint_callback])

if __name__ == "__main__":
    # モデルのトレーニング
    trainer.fit(model, train_loader, val_loader)

    # テストデータでモデルを評価
    trainer.test(model, test_loader)
