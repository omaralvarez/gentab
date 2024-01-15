from . import Tuner
from synthtab.generators import TVAE
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class TVAETuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        epochs = trial.suggest_int("epochs", 300, 600)
        batch_size_mult = trial.suggest_int("batch_size_mult", 16, 8192, step=2)
        compress_dims = (
            trial.suggest_int("comp_dims_in", 32, 512),
            trial.suggest_int("comp_dims_out", 32, 512),
        )
        decompress_dims = (
            trial.suggest_int("decomp_dims_in", 32, 512),
            trial.suggest_int("decomp_dims_out", 32, 512),
        )
        embedding_dim = trial.suggest_int("embedding_dim", 64, 1024)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)
        loss_factor = trial.suggest_int("loss_factor", 2, 16)
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

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
