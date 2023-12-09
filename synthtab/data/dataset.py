from synthtab.console import console, SPINNER, REFRESH
from synthtab import SEED

import pandas as pd
import numpy as np
import torch
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder
import os


class Dataset:
    def __init__(self, config) -> None:
        self.config = config
        self.X_gen = None
        self.y_gen = None

        with console.status(
            "Loading dataset {}...".format(self.config["name"]),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.X = pd.read_csv(self.config["path_X"])[:30000]
            self.y = pd.read_csv(self.config["path_y"])[:30000]

        console.print("✅ Dataset loaded...")

    def __str__(self) -> str:
        return self.config["name"]

    def save_to_disk(self, name) -> None:
        with console.status(
            "Saving dataset to {}...".format(self.config["save_path"]),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.X_gen.to_csv(
                os.path.join(self.config["save_path"], "X_gen_" + str(name) + ".csv"),
                index=False,
            )
            self.y_gen.to_csv(
                os.path.join(self.config["save_path"], "y_gen_" + str(name) + ".csv"),
                index=False,
            )

        console.print("✅ Dataset saved to {}...".format(self.config["save_path"]))

    def class_counts(self) -> int:
        return self.y[self.config["y_label"]].value_counts()

    def row_count(self) -> int:
        return len(self.X.index)

    def generated_class_counts(self) -> int:
        return self.y_gen.value_counts()

    def generated_row_count(self) -> int:
        return len(self.X_gen.index)

    def get_single_df(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y], axis=1)

    def set_split_result(self, data) -> None:
        self.X_gen = data.loc[:, data.columns != self.config["y_label"]]
        self.y_gen = data[self.config["y_label"]]

    def reduce_size(self, class_percentages) -> None:
        for cls, percent in class_percentages.items():
            self.reduce_uniformly_randomly(
                self.y[self.config["y_label"]] == cls, percent
            )

    def reduce_uniformly_randomly(self, condition, percentage) -> None:
        """
        Removes a random subset of rows from a DataFrame based on a condition.

        Parameters:
        - df: pandas DataFrame.
        - condition: A boolean series indicating which rows comply with the condition.
        - percentage: Percentage of rows to remove that comply with the condition.
        """
        # Identify rows that comply with the condition
        compliant_rows = self.y[condition]

        # Calculate the number of rows to remove
        n_remove = int(len(compliant_rows) * percentage)

        # Randomly select rows to remove
        rows_to_remove = compliant_rows.sample(n=n_remove, random_state=SEED).index

        # Remove the selected rows from the DataFrame
        self.X.drop(rows_to_remove, inplace=True)
        self.y.drop(rows_to_remove, inplace=True)

    def reduce_mem(self) -> None:
        """iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = self.X.memory_usage().sum() / 1024**2
        console.print("💾 Memory usage of dataframe is {:.2f} MB".format(start_mem))

        with console.status(
            "Reducing memory usage...", spinner=SPINNER, refresh_per_second=REFRESH
        ) as status:
            for col in self.X.columns:
                col_type = self.X[col].dtype

                if col_type != object:
                    c_min = self.X[col].min()
                    c_max = self.X[col].max()
                    if str(col_type)[:3] == "int":
                        if (
                            c_min > np.iinfo(np.int8).min
                            and c_max < np.iinfo(np.int8).max
                        ):
                            self.X[col] = self.X[col].astype(np.int8)
                        elif (
                            c_min > np.iinfo(np.int16).min
                            and c_max < np.iinfo(np.int16).max
                        ):
                            self.X[col] = self.X[col].astype(np.int16)
                        elif (
                            c_min > np.iinfo(np.int32).min
                            and c_max < np.iinfo(np.int32).max
                        ):
                            self.X[col] = self.X[col].astype(np.int32)
                        elif (
                            c_min > np.iinfo(np.int64).min
                            and c_max < np.iinfo(np.int64).max
                        ):
                            self.X[col] = self.X[col].astype(np.int64)
                    else:
                        if (
                            c_min > np.finfo(np.float16).min
                            and c_max < np.finfo(np.float16).max
                        ):
                            self.X[col] = self.X[col].astype(np.float16)
                        elif (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                        ):
                            self.X[col] = self.X[col].astype(np.float32)
                        else:
                            self.X[col] = self.X[col].astype(np.float64)
                else:
                    self.X[col] = self.X[col].astype("category")

        end_mem = self.X.memory_usage().sum() / 1024**2
        console.print("💾 Memory usage after optimization is: {:.2f} MB".format(end_mem))
        console.print(
            "✅ Memory usage reduced by {:.1f}%...".format(
                100 * (start_mem - end_mem) / start_mem
            )
        )
