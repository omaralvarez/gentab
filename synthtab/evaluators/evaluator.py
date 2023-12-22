from synthtab.console import console, SPINNER, REFRESH
from synthtab.utils import compute_accuracy, compute_f1_p_r
from synthtab import SEED

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    matthews_corrcoef,
)


class Evaluator:
    def __init__(self, generator) -> None:
        self.seed = SEED
        self.generator = generator
        self.accuracy = None
        self.macro = None
        self.weighted = None

    def __str__(self) -> str:
        return self.__name__

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

    def get_metrics(self, X, y, generator):
        with console.status(
            "Evaluating {} accuracy in {}...".format(generator, self.generator.dataset),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            X, y, X_test, _ = self.preprocess(
                X, y, self.generator.dataset.X_test, self.generator.dataset.y_test
            )

            predictions = self.postprocess(self.model.fit(X, y).predict(X_test))

            self.accuracy = self.compute_accuracy(
                self.generator.dataset.y_test, predictions
            )
            self.mcc = self.compute_mcc(self.generator.dataset.y_test, predictions)
            self.macro = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "macro"
            )
            self.weighted = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "weighted"
            )

            console.print(
                "🎯 {} Accuracy: {}".format(generator, round(self.accuracy * 100, 1))
            )
            console.print("🎯 {} MCC: {}".format(generator, round(self.mcc, 2)))

        console.print(
            "✅ Evaluation complete with {} for {} in {}...".format(
                self.__name__, generator, self.generator.dataset
            )
        )

    def evaluate_baseline(self) -> None:
        self.get_metrics(
            self.generator.dataset.X,
            self.generator.dataset.y,
            "Baseline",
        )

    def evaluate(self) -> None:
        self.get_metrics(
            self.generator.dataset.X_gen,
            self.generator.dataset.y_gen,
            self.generator,
        )
