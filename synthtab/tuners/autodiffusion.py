from . import Tuner
from synthtab.generators import AutoDiffusion
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class AutoDiffusionTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 200,
        max_epochs: int = 1000,
        min_batch: int = 64,
        max_batch: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(
            evaluator,
            trials,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            min_batch=min_batch,
            max_batch=max_batch,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_epochs = trial.suggest_int("n_epochs", self.min_epochs, self.max_epochs)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        diff_n_epochs = trial.suggest_int("diff_n_epochs", 200, 10000, step=2)
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
        )
        self.generator.generate()

        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
