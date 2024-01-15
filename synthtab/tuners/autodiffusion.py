from . import Tuner
from synthtab.generators import AutoDiffusion
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class AutoDiffusionTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_epochs = trial.suggest_int("n_epochs", 200, 10000)
        diff_n_epochs = trial.suggest_int("diff_n_epochs", 200, 10000)
        threshold = trial.suggest_float("threshold", 0.05, 0.05)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-5, log=True)
        lr = trial.suggest_float("lr", 2e-5, 2e-3, log=True)
        hidden_size = trial.suggest_int("hidden_size", 64, 512)
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_dims = trial.suggest_categorical(
            "hidden_dims",
            [
                (128, 256, 512, 256, 128),
                (256, 512, 1024, 512, 256),
                (512, 1024, 2048, 1024, 512),
            ],
        )
        sigma = trial.suggest_int("sigma", 10, 40)
        T = trial.suggest_int("T", 50, 400)

        # TODO Maybe add batch sizes
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
        )
        self.generator.generate()

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
