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
        self.__name__ = "LightGBM"
        self.model = lgb.LGBMClassifier(
            random_state=self.seed,
            *args,
            **kwargs,
        )
