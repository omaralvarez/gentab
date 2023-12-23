from . import Generator
from synthtab.utils import console, SPINNER, REFRESH

from imblearn.over_sampling import RandomOverSampler
from collections import Counter


class ROS(Generator):
    def __init__(self, dataset, sampling_strategy="auto", shrinkage=None) -> None:
        super().__init__(dataset)
        self.__name__ = "RandomOverSampler"
        self.shrinkage = shrinkage
        self.sampling_strategy = sampling_strategy

    def preprocess(self) -> None:
        super().preprocess()

    def train(self) -> None:
        super().train()

    def resample(self, n_samples) -> None:
        for cls, cnt in n_samples.items():
            n_samples[cls] += self.orig_counts[cls]

        ros = RandomOverSampler(
            random_state=self.seed,
            sampling_strategy=n_samples,
            shrinkage=self.shrinkage,
        )
        self.dataset.X_gen, self.dataset.y_gen = ros.fit_resample(
            self.dataset.X, self.dataset.y
        )

    def balance(self) -> None:
        ros = RandomOverSampler(
            random_state=self.seed,
            sampling_strategy=self.sampling_strategy,
            shrinkage=self.shrinkage,
        )
        self.dataset.X_gen, self.dataset.y_gen = ros.fit_resample(
            self.dataset.X, self.dataset.y
        )
