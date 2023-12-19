from synthtab.console import console, SPINNER, REFRESH
from synthtab.utils import compute_accuracy, compute_f1_p_r
from synthtab import SEED

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Evaluator:
    def __init__(self, generator) -> None:
        self.seed = SEED
        self.generator = generator

    def __str__(self) -> str:
        return self.__name__

    def compute_accuracy(self, y_true, y_pred):
        return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    def compute_f1_p_r(self, y_true, y_pred, average):
        return precision_recall_fscore_support(
            np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average=average
        )

    def evaluate(self) -> None:
        with console.status(
            "Evaluating original accuracy {}...".format(self.generator.dataset),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            predictions = self.model.fit(
                self.generator.dataset.X, self.generator.dataset.y
            ).predict(self.generator.dataset.X_test)

            accuracy = self.compute_accuracy(self.generator.dataset.y_test, predictions)
            macro = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "macro"
            )
            weighted = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "weighted"
            )

            console.print("🎯 Original Accuracy: {}".format(round(accuracy * 100, 1)))
            console.print(macro)
            console.print(weighted)

            status.update(
                "Evaluating {} accuracy...".format(self.__name__), spinner=SPINNER
            )

            predictions = self.model.fit(
                self.generator.dataset.X_gen, self.generator.dataset.y_gen
            ).predict(self.generator.dataset.X_test)

            accuracy = self.compute_accuracy(self.generator.dataset.y_test, predictions)
            macro = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "macro"
            )
            weighted = self.compute_f1_p_r(
                self.generator.dataset.y_test, predictions, "weighted"
            )

            console.print("🎯 Synthetic Accuracy: {}".format(round(accuracy * 100, 1)))
            console.print(macro)
            console.print(weighted)

        console.print("✅ Evaluation complete with {}...".format(self.__name__))
