from . import Generator
from synthtab.console import console, SPINNER, REFRESH

from imblearn.over_sampling import ADASYN as ada
from collections import Counter


class ADASYN(Generator):
    def __init__(self, dataset, n_neighbors=5, sampling_strategy="auto") -> None:
        super().__init__(dataset)
        self.__name__ = "ADASYN"
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy

    def resample(self, n_samples) -> None:
        for cls, cnt in n_samples.items():
            n_samples[cls] += self.orig_counts[cls]

        self.dataset.X_gen, self.dataset.y_gen = ada(
            random_state=self.seed,
            sampling_strategy=n_samples,
            n_neighbors=self.n_neighbors,
        ).fit_resample(self.dataset.X, self.dataset.y)

    def balance(self) -> None:
        if self.sampling_strategy == "minority":
            for _ in range(self.dataset.num_classes() - 2):
                self.dataset.X_gen, self.dataset.y_gen = ada(
                    random_state=self.seed,
                    sampling_strategy=self.sampling_strategy,
                    n_neighbors=self.n_neighbors,
                ).fit_resample(self.dataset.X, self.dataset.y)
        else:
            self.dataset.X_gen, self.dataset.y_gen = ada(
                random_state=self.seed,
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.n_neighbors,
            ).fit_resample(self.dataset.X, self.dataset.y)
