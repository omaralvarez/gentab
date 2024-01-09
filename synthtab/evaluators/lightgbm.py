from . import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import lightgbm as lgb


class LightGBM(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.model = lgb.LGBMClassifier(
            *args,
            random_state=self.seed,
            **kwargs,
        )
        self.callbacks = [lgb.log_evaluation(period=0)]

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X_test = self.dataset.encode_categories(X_test)

        return X, y, X_test, y_test
