from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH
from synthtab import SEED

import optuna


class Tuner:
    def __init__(self, evaluator: Evaluator, n_trials: int = 10) -> None:
        self.seed = SEED
        self.evaluator = evaluator
        self.generator = evaluator.generator
        self.n_trials = n_trials

    def __str__(self) -> str:
        return self.__name__

    def objective(self, trial: optuna.trial.Trial) -> float:
        pass

    def tune(self) -> None:
        self.pruner: optuna.pruners.BasePruner(optuna.pruners.NoPruner())

        self.study = optuna.create_study(direction="maximize", pruner=self.pruner)
        self.study.optimize(self.objective, n_trials=self.n_trials)

        console.print("Number of finished trials: {}".format(len(self.study.trials)))

        console.print("Best trial:")
        self.trial = self.study.best_trial

        console.print("  Value: {}".format(self.trial.value))

        console.print("  Params: ")
        for key, value in self.trial.params.items():
            console.print("    {}: {}".format(key, value))
