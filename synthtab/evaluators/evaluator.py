from synthtab.utils import ProgressBar, console, SPINNER, REFRESH
from synthtab import SEED

from typing import Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    matthews_corrcoef,
)


class Evaluator:
    def __init__(self, generator) -> None:
        self.seed = SEED
        self.generator = generator
        self.dataset = generator.dataset
        self.accuracy = None
        self.macro = None
        self.weighted = None
        self.callbacks = None

    def __str__(self) -> str:
        return self.__class__.__name__

    def preprocess(self, X, y, X_test, y_test):
        return X, y, X_test, y_test

    def postprocess(self, pred):
        return pred

    def compute_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def compute_mcc(self, y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    def compute_f1_p_r(self, y_true, y_pred, average):
        return precision_recall_fscore_support(y_true, y_pred, average=average)

    def compute_metrics(self, X, y, generator) -> None:
        with ProgressBar(indeterminate=True).progress as p:
            eval_task = p.add_task(
                "Evaluating {} accuracy in {}...".format(generator, self.dataset),
                total=None,
            )

            X, y, X_test, y_test = self.preprocess(
                X, y, self.dataset.X_test, self.dataset.y_test
            )
            if self.callbacks is not None:
                predictions = self.postprocess(
                    self.model.fit(X, y, callbacks=self.callbacks).predict(X_test)
                )
            else:
                predictions = self.postprocess(self.model.fit(X, y).predict(X_test))

            self.accuracy = self.compute_accuracy(self.dataset.y_test, predictions)
            self.mcc = self.compute_mcc(self.dataset.y_test, predictions)
            self.macro = self.compute_f1_p_r(self.dataset.y_test, predictions, "macro")
            self.weighted = self.compute_f1_p_r(
                self.dataset.y_test, predictions, "weighted"
            )

            console.print(
                "🎯 {} Accuracy: {}".format(generator, round(self.accuracy * 100, 1))
            )
            console.print("🎯 {} MCC: {}".format(generator, round(self.mcc, 2)))

        console.print(
            "✅ Evaluation complete with {} for {} in {}...".format(
                self, generator, self.dataset
            )
        )

    def evaluate_baseline(self) -> Tuple[float, float]:
        self.compute_metrics(
            self.dataset.X,
            self.dataset.y,
            "Baseline",
        )

        return self.accuracy, self.mcc

    def evaluate(self) -> Tuple[float, float]:
        self.compute_metrics(
            self.dataset.X_gen,
            self.dataset.y_gen,
            self.generator,
        )

        return self.accuracy, self.mcc
