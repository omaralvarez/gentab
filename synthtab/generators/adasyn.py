from . import Generator
from synthtab.console import console, SPINNER, REFRESH

from imblearn.over_sampling import ADASYN as ada
from collections import Counter


class ADASYN(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.__name__ = "ADASYN"

    def train(self) -> None:
        pass

    def sample(self) -> None:
        self.dataset.X_gen, self.dataset.y_gen = ada(
            sampling_strategy="auto"
        ).fit_resample(self.dataset.X, self.dataset.y)
