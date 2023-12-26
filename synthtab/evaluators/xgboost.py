from . import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

from xgboost import XGBClassifier


class XGBoost(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.model = XGBClassifier(
            *args,
            verbosity=0,
            random_state=self.seed,
            # enable_categorical=True,
            # tree_method="hist",
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        y = self.generator.dataset.label_encoder.transform(y)
        y_test = self.generator.dataset.label_encoder.transform(y_test)

        return X, y, X_test, y_test

    def postprocess(self, pred):
        return self.generator.dataset.label_encoder.inverse_transform(pred)
