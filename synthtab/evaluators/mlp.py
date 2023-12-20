from . import Evaluator
from synthtab.console import console, SPINNER, REFRESH

from typing_extensions import Self
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
import lightning as pl


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = torch.tensor(X.to_numpy())
        self.y = torch.tensor(y.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        batch_size: int = 8192,
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size

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

        if stage == "test" or stage is None:
            self.test = TabularDataset(self.X_test, self.y_test)

        if stage == "predict" or stage is None:
            self.predict = TabularDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)


class LightningMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class MLPClassifier:
    def __init__(self) -> None:
        self.model = LightningMLP()
        self.trainer = pl.Trainer(
            auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=5
        )

    def fit(self, X, y) -> Self:
        self.trainer.fit(self.model, TabularDataModule(X, y))
        return self

    def predict(self, X):
        self.trainer.predict(self.model, TabularDataModule(X))


class MLP(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.__name__ = "LightGBM"
        self.model = MLPClassifier()
