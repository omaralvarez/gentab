from . import Generator
from .ctabgplus.pipeline.data_preparation import DataPrep
from .ctabgplus.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class CTABGANPlus(Generator):
    def __init__(
        self,
        dataset,
        raw_csv_path="Real_Datasets/Adult.csv",
        test_ratio=0.20,
        categorical_columns=[],
        mixed_columns={},
        integer_columns=[],
        log_columns=[],
        general_columns=[],
        non_categorical_columns=[],
        problem_type={},
        epochs=100,
    ) -> None:
        super().__init__(dataset)
        self.__name__ = "CTABGANPlus"

        self.synthesizer = CTABGANSynthesizer(epochs=epochs)
        self.raw_df = pd.concat([self.dataset.X, self.dataset.y], axis=1)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.problem_type = problem_type

    def train(self) -> None:
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio,
        )

        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=self.problem_type,
        )

    def sample(self) -> None:
        sample = self.synthesizer.sample(len(self.raw_df))
        data_gen = self.data_prep.inverse_prep(sample)

        self.dataset.X_gen = data_gen.loc[
            :, data_gen.columns != self.dataset.config["y_label"]
        ]
        self.dataset.y_gen = data_gen[self.dataset.config["y_label"]]
