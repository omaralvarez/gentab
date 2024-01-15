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
        n_neighbors = trial.suggest_int(
            "n_neighbors", 2, self.dataset.min_class_count()
        )

        self.generator = ADASYN(self.dataset, n_neighbors=n_neighbors)
        self.generator.generate()

        acc, mcc = self.evaluator.evaluate()

        trial.set_user_attr("generator", self.generator)

        return mcc
