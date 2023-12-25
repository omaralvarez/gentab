from . import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

from sklearn.neighbors import KNeighborsClassifier


class KNN(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.model = KNeighborsClassifier(
            *args,
            **kwargs,
        )
