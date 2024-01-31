from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH
from synthtab import SEED

from pathlib import Path
import json
import os
import optuna


class Tuner:
    def __init__(
        self,
        evaluator: Evaluator,
        n_trials: int,
        min_epochs: int = 300,
        max_epochs: int = 800,
        min_batch: int = 512,
        max_batch: int = 4096,
    ) -> None:
        self.seed = SEED
        self.evaluator = evaluator
        self.generator = evaluator.generator
        self.dataset = evaluator.generator.dataset
        self.n_trials = n_trials
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.batch_sizes = [
            2**i
            for i in range(
                int(min_batch).bit_length() - 1, int(max_batch).bit_length() + 1
            )
            if 2**i >= min_batch and 2**i <= max_batch
        ]
        self.folder = "tuning"

    def __str__(self) -> str:
        return self.__class__.__name__

    def objective(self, trial: optuna.trial.Trial) -> float:
        pass

    def save_to_disk(self):
        self.dataset.save_to_disk(self.generator, self.evaluator)

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

        Path(self.folder).mkdir(parents=True, exist_ok=True)

        path = os.path.join(
            self.folder,
            str(self.dataset).lower()
            + "_"
            + str(self.generator).lower()
            + "_"
            + str(self.evaluator).lower()
            + ".json",
        )

        with open(path, "w") as fp:
            json.dump(self.trial.params, fp, indent=4)

        self.dataset = self.trial.user_attrs["dataset"]
