import os
import argparse
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
import optuna
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class HyperparameterTuner:
    def __init__(self, model_class, num_classes, train_dataset, val_dataset, accelerator, max_epochs=10, n_trials=20):
        self.model_class = model_class
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.accelerator = accelerator
        self.max_epochs = max_epochs
        self.n_trials = n_trials

    def objective(self, trial):
        # ハイパーパラメータの提案
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
        optimizer_class = getattr(torch.optim, optimizer_name)

        # モデルの初期化
        model = self.model_class(self.num_classes, lr=lr, optimizer_class=optimizer_class)

        # データローダーの設定
        train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

        # PyTorch Lightning Trainerの設定
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
            accelerator=self.accelerator
        )

        # 学習の実行
        trainer.fit(model, train_loader, val_loader)

        # 最後の検証損失を返す
        return trainer.callback_metrics["val_loss"].item()

    def tune(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return trial.params

def main(mode):
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

    data_dir = "./content/"
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, "dataimagegenerator_input"))

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

    num_classes = len(full_dataset.classes)

    if mode == "optuna":
        tuner = HyperparameterTuner(WeatherClassifier, num_classes, train_dataset, val_dataset, accelerator)
        best_params = tuner.tune()
        print(f"Best hyperparameters: {best_params}")
    else:
        # モデルのトレーニング
        model = WeatherClassifier(num_classes)
        trainer = Trainer(max_epochs=20, accelerator=accelerator, devices=1, logger=csv_logger, callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run the script in: 'train' or 'optuna'.")
    args = parser.parse_args()

    main(args.mode)