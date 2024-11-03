import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split

class WeatherClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(WeatherClassifier, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data) / len(labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data) / len(labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

# データの前処理
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "./content/dataimagegenerator_input/"
full_dataset = datasets.ImageFolder(
    os.path.join(data_dir, "train"), data_transforms["train"]
)

# データセットをランダムにシャッフルして分割
full_size = len(full_dataset)
val_size = int(0.2 * full_size)
test_size = int(0.2 * full_size)
train_size = full_size - test_size - val_size
train_val_dataset, test_dataset = random_split(full_dataset, [train_size + val_size, test_size])
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

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

trainer = Trainer(
    max_epochs=3, accelerator=accelerator, devices=1, logger=csv_logger
)

if __name__ == '__main__':
    # モデルのトレーニング
    trainer.fit(model, train_loader, val_loader)

    # テストデータでモデルを評価
    trainer.test(model, test_loader)
