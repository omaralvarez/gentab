from . import Generator
from synthtab.console import console, SPINNER, REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter


class TVAE(Generator):
    def __init__(
        self,
        dataset,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs=100,
        batch_size=8192,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        embedding_dim=128,
        cuda=True,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "TVAE"
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self.batch_size = batch_size
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.embedding_dim = embedding_dim
        self.cuda = cuda
        self.max_tries_per_batch = max_tries_per_batch

    def preprocess(self) -> None:
        self.data = self.dataset.get_single_df()

    def train(self) -> None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.data)

        self.synthesizer = TVAESynthesizer(
            metadata,  # required
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
            batch_size=self.batch_size,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            embedding_dim=self.embedding_dim,
            cuda=self.cuda,
        )
        self.synthesizer.fit(self.data)

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
            pd.concat([self.data, data_gen], ignore_index=True, sort=False)
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
            pd.concat([self.data, data_gen], ignore_index=True, sort=False)
        )
