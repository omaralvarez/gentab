from . import Generator
from synthtab.utils import console, SPINNER, REFRESH

import pandas as pd
from imblearn.over_sampling import ADASYN as ada
from collections import Counter


class ADASYN(Generator):
    def __init__(self, dataset, n_neighbors=5, sampling_strategy="minority") -> None:
        super().__init__(dataset)
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.X_ada = None

    def preprocess(self) -> None:
        self.X_ada = self.dataset.encode_categories(self.dataset.X)

    def train(self) -> None:
        super().train()

    def sample(self) -> pd.DataFrame:
        return super().sample()

    def resample(self, n_samples, append) -> None:
        if append:
            for cls, cnt in n_samples.items():
                n_samples[cls] += self.orig_counts[cls]

        self.dataset.X_gen, self.dataset.y_gen = ada(
            random_state=self.seed,
            sampling_strategy=n_samples,
            n_neighbors=self.n_neighbors,
        ).fit_resample(self.X_ada, self.dataset.y)

        self.dataset.X_gen = self.dataset.decode_categories(self.dataset.X_gen)

    def balance(self) -> None:
        if self.sampling_strategy == "minority":
            for _ in range(self.dataset.num_classes() - 1):
                self.dataset.X_gen, self.dataset.y_gen = ada(
                    random_state=self.seed,
                    sampling_strategy=self.sampling_strategy,
                    n_neighbors=self.n_neighbors,
                ).fit_resample(self.X_ada, self.dataset.y)
        else:
            self.dataset.X_gen, self.dataset.y_gen = ada(
                random_state=self.seed,
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.n_neighbors,
            ).fit_resample(self.X_ada, self.dataset.y)

        self.dataset.X_gen = self.dataset.decode_categories(self.dataset.X_gen)
