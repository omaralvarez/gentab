from . import Evaluator
from gentab.utils import console, SPINNER, REFRESH

from xgboost import XGBClassifier


class XGBoost(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator, **kwargs)
        self.model = XGBClassifier(
            *args,
            verbosity=0,
            random_state=self.seed,
            # enable_categorical=True,
            # tree_method="hist",
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X_test = self.dataset.encode_categories(X_test)

        y = self.dataset.encode_labels(y)
        y_test = self.dataset.encode_labels(y_test)

        return X, y, X_test, y_test

    def postprocess(self, pred):
        return self.dataset.decode_labels(pred)
