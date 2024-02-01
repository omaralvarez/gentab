from . import Tuner
from synthtab.generators import CTGAN
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class CTGANTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 300,
        max_epochs: int = 600,
        min_batch: int = 512,
        max_batch: int = 4096,
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
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        discriminator_dim = (
            trial.suggest_categorical("disc_dim_in", [32, 64, 128, 256, 512]),
            trial.suggest_categorical("disc_dim_out", [32, 64, 128, 256, 512]),
        )
        discriminator_decay = trial.suggest_float(
            "discriminator_decay", 1e-7, 1e-5, log=True
        )
        discriminator_lr = trial.suggest_float("discriminator_lr", 2e-5, 2e-3, log=True)
        discriminator_steps = trial.suggest_categorical(
            "discriminator_steps", [1, 2, 4, 8]
        )
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256, 512])
        generator_decay = trial.suggest_float("generator_decay", 1e-7, 1e-5, log=True)
        generator_dim = (
            trial.suggest_categorical("gen_dims_in", [32, 64, 128, 256, 512]),
            trial.suggest_categorical("gen_dims_out", [32, 64, 128, 256, 512]),
        )
        generator_lr = trial.suggest_float("discriminator_lr", 2e-5, 2e-3, log=True)

        log_frequency = trial.suggest_categorical("log_frequency", [True, False])
        pac = trial.suggest_categorical("pac", [2, 4, 8, 16, 32])

        self.generator = CTGAN(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            discriminator_dim=discriminator_dim,
            discriminator_decay=discriminator_decay,
            discriminator_lr=discriminator_lr,
            discriminator_steps=discriminator_steps,
            embedding_dim=embedding_dim,
            generator_decay=generator_decay,
            generator_dim=generator_dim,
            generator_lr=generator_lr,
            log_frequency=log_frequency,
            pac=pac,
        )
        self.generator.generate()

        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
