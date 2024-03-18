from . import Generator
from gentab.utils import console, DEVICE

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import pandas as pd
from collections import Counter


class CTGAN(Generator):
    def __init__(
        self,
        dataset,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs=300,
        batch_size=8000,
        discriminator_dim=(256, 256),
        discriminator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        embedding_dim=128,
        generator_decay=1e-6,
        generator_dim=(256, 256),
        generator_lr=2e-4,
        # https://github.com/sdv-dev/SDV/issues/1231 batch_size needs to be multiple of pac
        pac=10,
        log_frequency=True,
        cuda=True if DEVICE == "cuda" else False,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
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
        self.log_frequency = log_frequency
        self.cuda = cuda
        self.max_tries_per_batch = max_tries_per_batch

    def sample(self) -> pd.DataFrame:
        return super().sample()

    def preprocess(self) -> None:
        self.data = self.dataset.get_single_df()

    def train(self) -> None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.data)

        self.synthesizer = CTGANSynthesizer(
            metadata,  # required
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            discriminator_decay=self.discriminator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            embedding_dim=self.embedding_dim,
            generator_decay=self.generator_decay,
            generator_lr=self.generator_lr,
            pac=self.pac,
            log_frequency=self.log_frequency,
            cuda=self.cuda,
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
            conditions=conditions, batch_size=4096, max_tries_per_batch=4096
        )

        self.dataset.set_split_result(
            pd.concat([self.data, data_gen], ignore_index=True, sort=False)
        )
