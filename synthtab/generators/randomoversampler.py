from . import Generator
from synthtab.console import console,SPINNER,REFRESH

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

class ROS(Generator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        self.__name__ = 'RandomOverSampler'

    def train(self) -> None:
        pass

    def generate(self) -> None:
        ros = RandomOverSampler(random_state=0)
        self.dataset.X_gen, self.dataset.y_gen = ros.fit_resample(self.dataset.X, self.dataset.y)

