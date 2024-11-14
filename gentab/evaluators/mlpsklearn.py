from . import Evaluator
from gentab.utils import console, SPINNER, REFRESH

from sklearn.neural_network import MLPClassifier


class MLP(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator, **kwargs)
        self.model = MLPClassifier(
            *args,
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X = self.dataset.get_normalized_features(X)
        X_test = self.dataset.encode_categories(X_test)
        X_test = self.dataset.get_normalized_features(X_test)

        return X.fillna(0), y, X_test.fillna(0), y_test
