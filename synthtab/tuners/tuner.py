from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH
from synthtab import SEED

import optuna


class Tuner:
    def __init__(
        self,
        evaluator: Evaluator,
        n_trials: int = 10,
        min_epochs: int = 300,
        max_epochs: int = 800,
        min_batch: int = 512,
        max_batch: int = 4196,
    ) -> None:
        self.seed = SEED
        self.evaluator = evaluator
        self.generator = evaluator.generator
        self.dataset = evaluator.generator.dataset
        self.n_trials = n_trials
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.min_batch = min_batch
        self.max_batch = max_batch

    def __str__(self) -> str:
        return self.__class__.__name__

    def objective(self, trial: optuna.trial.Trial) -> float:
        pass

    def tune(self) -> None:
        # pruner: optuna.pruners.BasePruner(optuna.pruners.NopPruner())

        self.study = optuna.create_study(
            study_name=self,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        # TODO Test n_jobs = -1
        self.study.optimize(self.objective, n_trials=self.n_trials)

        console.print("Number of finished trials: {}".format(len(self.study.trials)))

        console.print("Best trial:")
        self.trial = self.study.best_trial

        console.print("  Value: {}".format(self.trial.value))

        console.print("  Params: ")
        for key, value in self.trial.params.items():
            console.print("    {}: {}".format(key, value))

        self.generator = self.trial.user_attrs["generator"]
