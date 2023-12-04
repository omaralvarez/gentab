from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from imblearn.over_sampling import SMOTE as sm
from collections import Counter

class SMOTE(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.__name__ = 'SMOTE'

    def train(self) -> None:
        pass

    def sample(self) -> None:
        self.dataset.X_gen, self.dataset.y_gen = sm().fit_resample(self.dataset.X, self.dataset.y)
