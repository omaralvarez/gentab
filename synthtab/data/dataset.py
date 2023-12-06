from synthtab.console import console, SPINNER, REFRESH

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
            # TODO Reomove range
            self.X = pd.read_csv(self.config["path_X"])  # [:8000]
            self.y = pd.read_csv(self.config["path_y"])  # [:8000]

        console.print("✅ Dataset loaded...")

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
