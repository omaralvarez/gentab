from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef
from . import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

from typing_extensions import Self
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
import lightning as pl
from lightning.pytorch.tuner import Tuner


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        # X_test: pd.DataFrame,
        # y_test: pd.DataFrame,
        batch_size: int = 8192,
        seed: int = 42,
    ):
        super().__init__()
        self.X = X
        self.y = y
        # self.X_test = X_test
        # self.y_test = y_test
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            train = TabularDataset(self.X, self.y)
            dataset_size = len(train)
            train_set_size = int(dataset_size * 0.9)
            valid_set_size = dataset_size - train_set_size

            self.train, self.val = random_split(
                train,
                [train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # if stage == "test" or stage is None:
        #     self.test = TabularDataset(self.X_test, self.y_test)

        if stage == "predict" or stage is None:
            self.predict = TabularDataset(self.X, self.y)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 4,
            pin_memory=True,
            # persistent_workers=True,
        )
        # return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 4,
            pin_memory=True,
            # persistent_workers=True,
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 4,
            pin_memory=True,
            # persistent_workers=True,
        )


class LightningMLP(pl.LightningModule):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self.ce = nn.CrossEntropyLoss()
        self.acc = MultilabelAccuracy(num_labels=num_classes)
        self.mcc = MultilabelMatthewsCorrCoef(num_labels=num_classes)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)

        loss = self.ce(y_hat, y)

        self.log_dict({"loss": loss, "acc": self.acc(y_hat, y), "mcc": self.mcc})

        return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     x = x.view(x.size(0), -1)
    #     y_hat = self.layers(x)
    #     loss = self.ce(y_hat, y)
    #     return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = torch.argmax(self(x), dim=1)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class MLPClassifier:
    def __init__(
        self,
        input_features,
        num_classes,
        *args,
        batch_size: int = 8192,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed

        torch.set_float32_matmul_precision("medium")

        self.model = LightningMLP(input_features, num_classes)
        self.trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            deterministic=True,
            max_epochs=100,
        )

    def fit(self, X, y) -> Self:
        self.datamodule = TabularDataModule(
            X,
            y,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        # self.tuner = Tuner(self.trainer)
        # self.batch_size = self.tuner.scale_batch_size(
        #     self.model, mode="power", datamodule=self.datamodule
        # )

        self.trainer.fit(self.model, self.datamodule)

        return self

    def predict(self, X):
        predictions = self.trainer.predict(
            self.model,
            TabularDataModule(
                X,
                torch.zeros(len(X.index)),
                batch_size=self.batch_size,
                seed=self.seed,
            ),
        )

        return torch.cat(predictions).numpy()


# TODO Improve robustness...
class MLP(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        batch_size: int = 32768,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.model = MLPClassifier(
            self.generator.dataset.num_features(),
            self.generator.dataset.num_classes(),
            *args,
            batch_size=batch_size,
            seed=self.seed,
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X_test = self.dataset.encode_categories(X_test)

        y = self.generator.dataset.label_encoder_ohe.transform(y)
        y_test = self.generator.dataset.label_encoder_ohe.transform(y_test)

        return X, y, X_test, y_test

    def postprocess(self, pred):
        return self.generator.dataset.label_encoder.inverse_transform(pred)
