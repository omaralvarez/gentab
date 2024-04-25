from . import Tuner
from gentab.generators import AutoDiffusion
from gentab.evaluators import Evaluator
from gentab.utils import console, SPINNER, REFRESH

import optuna


class AutoDiffusionTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 500,
        max_epochs: int = 10000,
        min_batch: int = 64,
        max_batch: int = 8192,
        max_tries_per_batch: int = 8192,
        timeout: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            evaluator,
            trials,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            min_batch=min_batch,
            max_batch=max_batch,
            max_tries_per_batch=max_tries_per_batch,
            timeout=timeout,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_epochs = trial.suggest_int("n_epochs", self.min_epochs, self.max_epochs)
        diff_n_epochs = trial.suggest_int(
            "diff_n_epochs", self.min_epochs, self.max_epochs
        )
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        threshold = trial.suggest_float("threshold", 0.005, 0.02)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True)
        lr = trial.suggest_float("lr", 2e-5, 2e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        num_layers = trial.suggest_categorical("num_layers", [2, 4, 8])
        hidden_dims = trial.suggest_categorical(
            "hidden_dims",
            [
                (128, 256, 512, 256, 128),
                (256, 512, 1024, 512, 256),
                # (512, 1024, 2048, 1024, 512),
            ],
        )
        sigma = trial.suggest_int("sigma", 10, 40)
        T = trial.suggest_int("T", 50, 200, step=2)

        self.generator = AutoDiffusion(
            self.dataset,
            n_epochs=n_epochs,
            diff_n_epochs=diff_n_epochs,
            threshold=threshold,
            weight_decay=weight_decay,
            lr=lr,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            sigma=sigma,
            T=T,
            batch_size=batch_size,
            max_tries_per_batch=self.max_tries_per_batch,
        )
        self.generator.generate()

        self.store_data(trial)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
