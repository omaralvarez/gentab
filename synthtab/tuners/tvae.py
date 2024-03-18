from . import Tuner
from synthtab.generators import TVAE
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class TVAETuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 300,
        max_epochs: int = 800,
        min_batch: int = 512,
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
        epochs = trial.suggest_int("epochs", self.min_epochs, self.max_epochs)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        compress_dims = (
            trial.suggest_categorical("comp_dims_in", [32, 64, 128, 256, 512]),
            trial.suggest_categorical("comp_dims_out", [32, 64, 128, 256, 512]),
        )
        decompress_dims = (
            trial.suggest_categorical("decomp_dims_in", [32, 64, 128, 256, 512]),
            trial.suggest_categorical("decomp_dims_out", [32, 64, 128, 256, 512]),
        )
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 512])
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)
        loss_factor = trial.suggest_categorical("loss_factor", [2, 4, 8, 16])
        pac = trial.suggest_categorical("pac", [2, 4, 8, 16, 32])

        self.generator = TVAE(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            loss_factor=loss_factor,
            pac=pac,
            max_tries_per_batch=self.max_tries_per_batch,
        )
        self.generator.generate()

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
