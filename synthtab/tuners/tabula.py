from . import Tuner
from synthtab.generators import Tabula
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class TabulaTuner(Tuner):
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
        encode_categories = trial.suggest_categorical(
            "encode_categories", [False, True]
        )
        middle_padding = trial.suggest_categorical("middle_padding", [False, True])
        random_initialization = trial.suggest_categorical(
            "random_initialization", [False, True]
        )

        self.generator = Tabula(
            self.dataset,
            epochs=epochs,
            encode_categories=encode_categories,
            middle_padding=middle_padding,
            random_initialization=random_initialization,
            batch_size=batch_size,
            temperature=temperature,
            k=k,
        )
        self.generator.generate()

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
