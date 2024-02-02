from . import Tuner
from synthtab.generators import CTABGANPlus
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class CTABGANPlusTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 300,
        max_epochs: int = 800,
        min_batch: int = 512,
        max_batch: int = 16384,
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
        test_ratio = trial.suggest_float("test_ratio", 0.1, 0.3)
        epochs = trial.suggest_int("epochs", self.min_epochs, self.max_epochs)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        class_dim = trial.suggest_categorical(
            "class_dim",
            [
                (128, 256, 256, 128),
                (256, 256, 256, 256),
                (512, 512, 512, 512),
            ],
        )
        random_dim = trial.suggest_int("random_dim", 50, 200)
        num_channels = trial.suggest_categorical("num_channels", [16, 32, 64, 128])
        l2scale = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)

        self.generator = CTABGANPlus(
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

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate()

        return mcc
