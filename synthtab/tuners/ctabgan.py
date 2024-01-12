from . import Tuner
from synthtab.generators import CTABGAN
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class CTABGANTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        test_ratio = trial.suggest_float("test_ratio", 0.1, 0.3)
        epochs = trial.suggest_int("epochs", 100, 800)
        class_dim = trial.suggest_categorical(
            "class_dim",
            [
                (256, 256, 256, 256),
                (512, 512, 512, 512),
                (128, 256, 256, 128),
            ],
        )
        random_dim = trial.suggest_int("random_dim", 50, 400)
        num_channels = trial.suggest_int("num_channels", 16, 128)
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 512, 16384)

        self.generator = CTABGAN(
            self.dataset,
            test_ratio=test_ratio,
            epochs=epochs,
            class_dim=class_dim,
            num_channels=num_channels,
            random_dim=random_dim,
            l2scale=l2scale,
            batch_size=batch_size,
        )
        self.generator.generate()

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
