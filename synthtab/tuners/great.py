from . import Tuner
from synthtab.generators import GReaT
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class GReaTTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        epochs = trial.suggest_int("epochs", 200, 800)
        batch_size = trial.suggest_int("batch_size", 4, 8)
        temperature = trial.suggest_float("temperature", 0.3, 1.0, log=True)
        k = trial.suggest_int("k", 50, 200)

        self.generator = GReaT(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
            temperature=temperature,
            k=k,
        )
        self.generator.generate()

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
