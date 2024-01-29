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
        min_batch: int = 16,
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
        epochs = trial.suggest_int("epochs", self.min_epochs, self.max_epochs)
        batch_size_mult = trial.suggest_int(
            "batch_size_mult", self.min_batch, self.max_batch, step=2
        )
        compress_dims = (
            trial.suggest_int("comp_dims_in", 32, 512, step=2),
            trial.suggest_int("comp_dims_out", 32, 512, step=2),
        )
        decompress_dims = (
            trial.suggest_int("decomp_dims_in", 32, 512, step=2),
            trial.suggest_int("decomp_dims_out", 32, 512, step=2),
        )
        embedding_dim = trial.suggest_int("embedding_dim", 64, 1024, step=2)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)
        loss_factor = trial.suggest_int("loss_factor", 2, 16, step=2)
        pac = trial.suggest_int("pac", 2, 256, step=2)

        self.generator = TVAE(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size_mult * pac,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            loss_factor=loss_factor,
            pac=pac,
        )
        self.generator.generate()

        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
