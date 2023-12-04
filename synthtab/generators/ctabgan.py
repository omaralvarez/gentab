from . import Generator
from .ctabg.pipeline.data_preparation import DataPrep 
from .ctabg.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 
from synthtab.console import console,SPINNER,REFRESH

import pandas as pd
import time

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
    def __init__(self,
                 dataset,
                 test_ratio = 0.10,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 integer_columns = [],
                 problem_type= {'Classification': '#play'},
                 epochs = 150) -> None:
        super().__init__(dataset)
        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(epochs = epochs)
        self.raw_df = pd.concat([self.dataset.X, self.dataset.y], axis=1)
        self.test_ratio = test_ratio
        self.categorical_columns = [self.dataset.config['y_label']]
        self.log_columns = log_columns
        self.mixed_columns = dict([(c,[0.0]) for c in self.dataset.X.columns])
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        
    def train(self) -> None:
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
        mixed = self.data_prep.column_types["mixed"], type=self.problem_type)

    def sample(self) -> None:
        sample = self.synthesizer.sample(len(self.raw_df)) 
        data_gen = self.data_prep.inverse_prep(sample)
        
        self.dataset.X_gen = data_gen.loc[:, data_gen.columns != self.dataset.config['y_label']]
        self.dataset.y_gen = data_gen[self.dataset.config['y_label']]
