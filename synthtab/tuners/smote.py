from . import Tuner
from synthtab.generators import SMOTE
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class SMOTETuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator, trials)

    def objective(self, trial: optuna.trial.Trial) -> float:
        k_neighbors = trial.suggest_int(
            "k_neighbors", 2, self.dataset.min_class_count(), log=True
        )

        self.generator = SMOTE(self.dataset, k_neighbors=k_neighbors)
        self.generator.generate()

        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
