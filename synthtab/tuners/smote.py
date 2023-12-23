from . import Tuner
from synthtab.generators import SMOTE
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class SMOTETuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)
        self.__name__ = "SMOTETuner"

    def objective(self, trial: optuna.trial.Trial) -> float:
        k_neighbors = trial.suggest_int("k_neighbors", 2, 16384)

        self.generator = SMOTE(k_neighbors=k_neighbors)
        self.generator.generate()

        acc, mcc = self.evaluator.evaluate()

        return mcc
