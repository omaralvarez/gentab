from . import Tuner
from gentab.generators import ADASYN
from gentab.evaluators import Evaluator
from gentab.utils import console, SPINNER, REFRESH

import optuna


class ADASYNTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        timeout: int = None,
        **kwargs,
    ) -> None:
        super().__init__(evaluator, trials, timeout=timeout)

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_neighbors = trial.suggest_int(
            "n_neighbors", 2, self.dataset.min_class_count(), log=True
        )

        self.generator = ADASYN(self.dataset, n_neighbors=n_neighbors)
        self.generator.generate()

        self.store_data(trial)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
