from . import Tuner
from synthtab.generators import GaussianCopula
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class GaussianCopulaTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        max_tries_per_batch: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(evaluator, trials, max_tries_per_batch=max_tries_per_batch)

    def objective(self, trial: optuna.trial.Trial) -> float:
        default_distribution = trial.suggest_categorical(
            "default_distribution",
            ["norm", "beta", "truncnorm", "uniform", "gamma", "gaussian_kde"],
        )

        self.generator = GaussianCopula(
            self.dataset,
            default_distribution=default_distribution,
            max_tries_per_batch=self.max_tries_per_batch,
        )
        self.generator.generate()

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
