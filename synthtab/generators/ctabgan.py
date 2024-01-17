from . import Generator
from .ctabg.pipeline.data_preparation import DataPrep
from .ctabg.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from synthtab.utils import console

import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class CTABGAN(Generator):
    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables Constructor:
    1) random_dim -> size of the noise vector fed to the generator
    2) class_dim -> tuple containing dimensionality of hidden layers for the classifier network
    3) num_channels -> no. of channels for deciding respective hidden layers of discriminator and generator networks
    4) l2scale -> parameter to decide strength of regularization of the network based on constraining l2 norm of weights
    5) batch_size -> no. of records to be processed in each mini-batch of training
    6) epochs -> no. of epochs to train the model

    Variables preprocess and generation:
    * Column names can appear in multiple parameters: e.g. capital-gain, integer_columns & mixed_columns
    1) test_ratio -> parameter to choose ratio of size of test to train data
    2) categorical_columns -> list of column names with a categorical distribution (including label)
    3) log_columns -> list of column names with a skewed exponential distribution
       e.g. amount (credit card transactions), most transactions have small amounts,
       very small amount of transactions with very high amounts, fig1b in paper
    4) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution
       e.g. mortgage, it can be no mortgage (0) or mortgage (any value), Fig1a in paper
    5) integer_columns -> list of numeric column names without floating numbers
    6) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    """

    def __init__(
        self,
        dataset,
        test_ratio=0.20,
        epochs=300,
        class_dim=(256, 256, 256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=8192,
        max_tries_per_batch=4096,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        # TODO something here makes the nex generator crash when calling set split results KeyError: '#play'
        self.raw_df = self.dataset.get_single_df()
        self.test_ratio = test_ratio
        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self.categorical_columns = (
            self.config["categorical_columns"]
            + self.config["binary_columns"]
            + [self.config["y_label"]]
        )
        self.log_columns = self.config[str(self)]["log_columns"]
        self.mixed_columns = self.config[str(self)]["mixed_columns"]
        self.integer_columns = self.config["integer_columns"]
        if self.config["task_type"] in ["multiclass", "binary"]:
            self.problem_type = {"Classification": dataset.config["y_label"]}
        else:
            self.problem_type = {"Regression": dataset.config["y_label"]}

        self.synthesizer = CTABGANSynthesizer(
            class_dim=self.class_dim,
            random_dim=self.random_dim,
            num_channels=self.num_channels,
            l2scale=self.l2scale,
            batch_size=self.batch_size,
            epochs=epochs,
        )

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

    def sample(self) -> pd.DataFrame:
        sample = self.synthesizer.sample(self.batch_size)
        return self.data_prep.inverse_prep(sample)
