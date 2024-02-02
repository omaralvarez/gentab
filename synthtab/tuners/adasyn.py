from . import Tuner
from synthtab.generators import ADASYN
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna
import copy


class ADASYNTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator, trials)

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_neighbors = trial.suggest_int(
            "n_neighbors", 2, self.dataset.min_class_count(), log=True
        )

        self.generator = ADASYN(self.dataset, n_neighbors=n_neighbors)
        self.generator.generate()

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", copy.copy(self.dataset))

        acc, mcc = self.evaluator.evaluate()

        return mcc
