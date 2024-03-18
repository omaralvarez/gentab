from . import Tuner
from gentab.generators import GReaT
from gentab.evaluators import Evaluator
from gentab.utils import console, SPINNER, REFRESH

import optuna


class GReaTTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_epochs: int = 200,
        max_epochs: int = 800,
        min_batch: int = 4,
        max_batch: int = 8,
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
        temperature = trial.suggest_float("temperature", 0.3, 1.0)
        k = trial.suggest_int("k", 50, 200, step=2)

        self.generator = GReaT(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            temperature=temperature,
            k=k,
            max_tries_per_batch=self.max_tries_per_batch,
        )
        self.generator.generate()

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
