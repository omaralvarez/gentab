from . import Tuner
from synthtab.generators import CopulaGAN
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class CopulaGANTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        default_distribution = trial.suggest_categorical(
            "default_distribution",
            ["norm", "beta", "truncnorm", "uniform", "gamma", "gaussian_kde"],
        )
        epochs = trial.suggest_int("epochs", 300, 600)
        batch_size_mult = trial.suggest_int("batch_size_mult", 16, 8192, step=2)
        discriminator_dim = (
            trial.suggest_int("disc_dim_in", 32, 512),
            trial.suggest_int("disc_dim_out", 32, 512),
        )
        discriminator_decay = trial.suggest_float(
            "discriminator_decay", 1e-7, 1e-5, log=True
        )
        discriminator_lr = trial.suggest_float("discriminator_lr", 2e-5, 2e-3, log=True)
        discriminator_steps = trial.suggest_int("discriminator_steps", 1, 16)
        embedding_dim = trial.suggest_int("embedding_dim", 64, 1024)
        generator_decay = trial.suggest_float("generator_decay", 1e-7, 1e-5, log=True)
        generator_dim = (
            trial.suggest_int("gen_dims_in", 32, 512),
            trial.suggest_int("gen_dims_out", 32, 512),
        )
        generator_lr = trial.suggest_float("discriminator_lr", 2e-5, 2e-3, log=True)

        log_frequency = trial.suggest_categorical("log_frequency", [True, False])
        pac = trial.suggest_int("pac", 2, 256, step=2)

        self.generator = CopulaGAN(
            self.dataset,
            default_distribution=default_distribution,
            epochs=epochs,
            batch_size=batch_size_mult * pac,
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

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
