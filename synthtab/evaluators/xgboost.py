from . import Evaluator
from synthtab.console import console, SPINNER, REFRESH

from xgboost import XGBClassifier


class XGBoost(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(generator)
        self.__name__ = "XGBoost"
        self.model = XGBClassifier(
            random_state=self.seed,
            *args,
            **kwargs,
        )
