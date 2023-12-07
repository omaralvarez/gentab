from . import Generator
from .ctabg.pipeline.data_preparation import DataPrep
from .ctabg.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from synthtab.console import console, SPINNER, REFRESH

import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class CTABGAN(Generator):
    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution
    6) integer_columns -> list of numeric column names without floating numbers
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data

    """

    # TODO Structure to have common problem types, automatic processing and so on
    def __init__(
        self,
        dataset,
        test_ratio=0.20,
        categorical_columns=[],
        log_columns=[],
        mixed_columns={},
        integer_columns=[],
        problem_type={},
        epochs=100,
        batch_size=8192,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "CTABGAN"

        self.synthesizer = CTABGANSynthesizer(epochs=epochs)
        self.raw_df = self.dataset.get_single_df()
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.batch_size = batch_size
        self.max_tries_per_batch = max_tries_per_batch

    def preprocess(self) -> None:
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio,
        )

    def train(self) -> None:
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            type=self.problem_type,
        )

    def sample(self, batch_size) -> pd.DataFrame:
        sample = self.synthesizer.sample(batch_size)
        return self.data_prep.inverse_prep(sample)

    def resample(self, n_samples) -> None:
        data_gen = self.raw_df

        for _ in range(self.max_tries_per_batch):
            gen = self.sample(self.batch_size)

            for cls, cnt in n_samples.items():
                if cnt > 0:
                    filtered = gen[gen[self.dataset.config["y_label"]] == cls]

                    count = len(filtered.index)
                    if count > cnt:
                        n_samples[cls] = 0
                        filtered = filtered.sample(n=cnt)
                    else:
                        n_samples[cls] = cnt - count

                    data_gen = pd.concat(
                        [data_gen, filtered], ignore_index=True, sort=False
                    )

            if sum(n_samples.values()) == 0:
                break

        self.dataset.set_split_result(data_gen)

    def balance(self) -> None:
        data_gen = self.raw_df

        for _ in range(self.max_tries_per_batch):
            sample = self.synthesizer.sample(self.batch_size)
            gen = self.data_prep.inverse_prep(sample)

            for cls, cnt in self.counts.items():
                if cnt > 0:
                    filtered = gen[gen[self.dataset.config["y_label"]] == cls]

                    count = len(filtered.index)
                    if count > cnt:
                        self.counts[cls] = 0
                        filtered = filtered.sample(n=cnt)
                    else:
                        self.counts[cls] = cnt - count

                    data_gen = pd.concat(
                        [data_gen, filtered], ignore_index=True, sort=False
                    )

            if self.counts.max() < 1:
                break

        self.dataset.set_split_result(data_gen)
