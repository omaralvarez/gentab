from . import Tuner
from synthtab.generators import ADASYN
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class ADASYNTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_neighbors = trial.suggest_int("n_neighbors", 2, 16384)

        self.generator = ADASYN(self.dataset, n_neighbors=n_neighbors)
        self.generator.generate()

        acc, mcc = self.evaluator.evaluate()

        return mcc
