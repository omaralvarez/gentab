from . import Generator
from tabgen.utils import console

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
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.locales = locales
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch

    def sample(self) -> pd.DataFrame:
        return super().sample()

    def preprocess(self) -> None:
        self.data = self.dataset.get_single_df()

    def train(self) -> None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.data)

        self.synthesizer = GaussianCopulaSynthesizer(
            metadata,  # required
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            default_distribution=self.default_distribution,
            numerical_distributions=self.numerical_distributions,
            locales=self.locales,
        )
        self.synthesizer.fit(self.data)

    def resample(self, n_samples, append) -> None:
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

        if append:
            self.dataset.set_split_result(
                pd.concat([self.data, data_gen], ignore_index=True, sort=False)
            )
        else:
            self.dataset.set_split_result(data_gen)

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
            pd.concat([self.data, data_gen], ignore_index=True, sort=False)
        )
