from . import Generator
from synthtab.console import console, SPINNER, REFRESH

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter


class CopulaGAN(Generator):
    def __init__(
        self,
        dataset,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs=100,
        batch_size=8192,
        discriminator_dim=(256, 256),
        discriminator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        embedding_dim=128,
        generator_decay=1e-6,
        generator_dim=(256, 256),
        generator_lr=2e-4,
        # TODO https://github.com/sdv-dev/SDV/issues/1231 maybe set batch size to 10*x
        pac=10,
        cuda=True,
        locales=["en_US"],
        numerical_distributions=None,
        default_distribution="beta",
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "CopulaGAN"
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.discriminator_decay = discriminator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.embedding_dim = embedding_dim
        self.generator_decay = generator_decay
        self.generator_lr = generator_lr
        self.pac = pac
        self.cuda = cuda
        self.locales = locales
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self.max_tries_per_batch = max_tries_per_batch

    def train(self) -> None:
        data = pd.concat([self.dataset.X, self.dataset.y], axis=1)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)

        self.synthesizer = CopulaGANSynthesizer(
            metadata,  # required
            enforce_min_max_values=True,
            enforce_rounding=False,
            default_distribution="gaussian_kde",
            epochs=100,
            batch_size=4000,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            embedding_dim=128,
            # TODO Same as CTGAN
            pac=10,
            cuda=True,
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
