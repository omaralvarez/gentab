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
        test_ratio=0.20,
        epochs=300,
        batch_size=8192,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        # TODO Use the rest of the constructor options
        self.synthesizer = CTABGANSynthesizer(epochs=epochs)
        self.raw_df = self.dataset.get_single_df()
        self.test_ratio = test_ratio
        self.categorical_columns = (
            self.config["categorical_columns"]
            + self.config["binary_columns"]
            + [self.config["y_label"]]
        )
        self.integer_columns = self.config["integer_columns"]
        self.log_columns = self.config[str(self)]["log_columns"]
        self.mixed_columns = self.config[str(self)]["mixed_columns"]
        self.general_columns = self.config[str(self)]["general_columns"]
        self.non_categorical_columns = self.config[str(self)]["non_categorical_columns"]
        if self.config["task_type"] in ["multiclass", "binary"]:
            self.problem_type = {"Classification": dataset.config["y_label"]}
        else:
            self.problem_type = {"Regression": dataset.config["y_label"]}

    def preprocess(self) -> None:
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

    def train(self) -> None:
        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=self.problem_type,
        )

    def sample(self) -> pd.DataFrame:
        sample = self.synthesizer.sample(self.batch_size)
        return self.data_prep.inverse_prep(sample)
