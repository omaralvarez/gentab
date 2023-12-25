from . import Tuner
from synthtab.generators import GaussianCopula
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class GaussianCopulaTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        default_distribution = trial.suggest_categorical(
            "default_distribution",
            ["norm", "beta", "truncnorm", "uniform", "gamma", "gaussian_kde"],
        )

        self.generator = GaussianCopula(
            self.dataset, default_distribution=default_distribution
        )
        self.generator.generate()

        acc, mcc = self.evaluator.evaluate()

        return mcc
