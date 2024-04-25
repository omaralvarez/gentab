from . import Evaluator
from gentab.utils import console, SPINNER, REFRESH

from catboost import CatBoostClassifier


class CatBoost(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator, **kwargs)

        self.model = CatBoostClassifier(
            *args,
            random_state=self.seed,
            silent=True,
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X_test = self.dataset.encode_categories(X_test)

        return X, y, X_test, y_test
