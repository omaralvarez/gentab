from . import Tuner
from gentab.generators import GaussianCopula
from gentab.evaluators import Evaluator
from gentab.utils import console, SPINNER, REFRESH

import optuna


class GaussianCopulaTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        max_tries_per_batch: int = 8192,
        timeout: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            evaluator, trials, max_tries_per_batch=max_tries_per_batch, timeout=timeout
        )

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

        self.store_data(trial)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
