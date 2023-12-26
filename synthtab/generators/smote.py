from . import Generator
from synthtab.utils import console, SPINNER, REFRESH

import pandas as pd
from imblearn.over_sampling import SMOTE as sm
from imblearn.over_sampling import SMOTEN as smn
from collections import Counter


class SMOTE(Generator):
    def __init__(self, dataset, k_neighbors=5, sampling_strategy="auto") -> None:
        super().__init__(dataset)
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy

    def preprocess(self) -> None:
        super().preprocess()

    def train(self) -> None:
        super().train()

    def sample(self) -> pd.DataFrame:
        return super().sample()

    def resample(self, n_samples) -> None:
        for cls, cnt in n_samples.items():
            n_samples[cls] += self.orig_counts[cls]

        self.dataset.X_gen, self.dataset.y_gen = sm(
            random_state=self.seed,
            sampling_strategy=n_samples,
            k_neighbors=self.k_neighbors,
        ).fit_resample(self.dataset.X, self.dataset.y)

    def balance(self) -> None:
        console.print(self.dataset.X.columns)
        if len(self.dataset.X._get_numeric_data().columns) != len(
            self.dataset.X.columns
        ):
            self.dataset.X_gen, self.dataset.y_gen = smn(
                random_state=self.seed,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
            ).fit_resample(self.dataset.X, self.dataset.y)
        else:
            self.dataset.X_gen, self.dataset.y_gen = sm(
                random_state=self.seed,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.k_neighbors,
            ).fit_resample(self.dataset.X, self.dataset.y)
