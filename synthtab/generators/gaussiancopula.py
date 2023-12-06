from . import Generator
from synthtab.console import console, SPINNER, REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter


class GaussianCopula(Generator):
    def __init__(
        self,
        dataset,
        enforce_min_max_values=True,
        enforce_rounding=False,
        locales=["en_US"],
        numerical_distributions=None,
        default_distribution="beta",
        batch_size=8192,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "GaussianCopula"
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.locales = locales
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch

    def train(self) -> None:
        data = pd.concat([self.dataset.X, self.dataset.y], axis=1)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        self.synthesizer = GaussianCopulaSynthesizer(
            metadata,  # required
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            default_distribution=self.default_distribution,
            numerical_distributions=self.numerical_distributions,
            locales=self.locales,
        )
        self.synthesizer.fit(data)

    def resample(self, n_samples) -> None:
        conditions = []
        for cls, cnt in n_samples.items():
            conditions.append(
                Condition(
                    num_rows=cnt, column_values={self.dataset.config["y_label"]: cls}
                )
            )

        data_gen = self.synthesizer.sample_from_conditions(
            conditions=conditions,
            batch_size=self.batch_size,
            max_tries_per_batch=self.max_tries_per_batch,
        )

        self.dataset.set_split_result(
            pd.concat(
                [self.dataset.get_single_df(), data_gen], ignore_index=True, sort=False
            )
        )

    def balance(self) -> None:
        conditions = []
        for cls, cnt in self.counts.items():
            conditions.append(
                Condition(
                    num_rows=cnt, column_values={self.dataset.config["y_label"]: cls}
                )
            )

        data_gen = self.synthesizer.sample_from_conditions(
            conditions=conditions,
            batch_size=self.batch_size,
            max_tries_per_batch=self.max_tries_per_batch,
        )

        self.dataset.set_split_result(
            pd.concat(
                [self.dataset.get_single_df(), data_gen], ignore_index=True, sort=False
            )
        )
