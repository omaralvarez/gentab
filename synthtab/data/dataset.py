from synthtab.utils import console, SPINNER, REFRESH
from synthtab import SEED

import pandas as pd
import numpy as np
import torch
from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.datasets import fetch_datasets
from ucimlrepo import fetch_ucirepo
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
            if not self.config.exists("download"):
                self.X = pd.read_csv(self.config["path_X"])
                self.y = pd.read_csv(self.config["path_y"])
                self.X_test = pd.read_csv(self.config["path_X_test"])
                self.y_test = pd.read_csv(self.config["path_y_test"])
            else:
                if self.config["download"] == "ucimlrepo":
                    self.download_uci()
                else:
                    self.download_imb()

            self.encode_labels()
            self.get_categories()

        console.print("✅ Dataset loaded...")

    def __str__(self) -> str:
        return self.config["name"]

    def download_imb(self) -> None:
        data = fetch_datasets(
            filter_data=(self.config["name"],),
            data_home=self.config["save_path"],
            random_state=SEED,
        )[self.config["name"]]

        features = pd.DataFrame(
            data=data.data,
            columns=["f" + str(i + 1) for i in range(data.data.shape[1])],
        )
        labels = pd.DataFrame({self.config["y_label"]: data.target})

        self.X, self.X_test, self.y, self.y_test = train_test_split(
            features,
            labels,
            test_size=self.config["test_size"],
            random_state=SEED,
            stratify=labels,
        )

        self.config["binary_columns"] = self.X.columns.values.tolist()
        self.config["categorical_columns"] = []
        self.config["integer_columns"] = []

        # train_ratio = 0.75
        # validation_ratio = 0.15
        # test_ratio = 0.10

        # # train is now 75% of the entire data set
        # x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

        # # test is now 10% of the initial data set
        # # validation is now 15% of the initial data set
        # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

        # print(x_train, x_val, x_test)

    def download_uci(self) -> None:
        uci = fetch_ucirepo(name=self.config["name"])

        # metadata
        # console.print(uci.metadata)
        console.print(uci.variables)

        uci.data.features.fillna("Missing", inplace=True)

        self.X, self.X_test, self.y, self.y_test = train_test_split(
            uci.data.features,
            uci.data.targets,
            test_size=self.config["test_size"],
            random_state=SEED,
            stratify=uci.data.targets,
        )

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

    def load_from_disk(self, name) -> None:
        with console.status(
            "Loading {} dataset...".format(name),
            spinner=SPINNER,
            refresh_per_second=REFRESH,
        ) as status:
            self.X_gen = pd.read_csv(
                os.path.join(self.config["save_path"], "X_gen_" + str(name) + ".csv")
            )
            self.y_gen = pd.read_csv(
                os.path.join(self.config["save_path"], "y_gen_" + str(name) + ".csv")
            )

        console.print("✅ {} dataset loaded...".format(name))

    def get_categories(self) -> None:
        self.cats = self.config["categorical_columns"] + self.config["binary_columns"]
        self.X_cats = self.X[self.cats].copy()
        self.X_cats[self.cats] = self.X_cats[self.cats].apply(
            lambda col: pd.Categorical(col)
        )

    def encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config["download"] == "imbalanced":
            return df
        else:
            X_enc = df.copy()
            X_enc[self.cats] = X_enc[self.cats].apply(
                lambda col: pd.Categorical(col).codes
            )

        return X_enc

    def decode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config["download"] == "imbalanced":
            return df
        else:
            cat_columns = self.X_cats.select_dtypes(["category"]).columns
            df[cat_columns] = df[cat_columns].apply(
                lambda col: pd.Categorical.from_codes(
                    col, self.X_cats[col.name].cat.categories
                )
            )

        return df

    def merge_classes(self, merge: dict[str, list[str]]) -> None:
        for cls, labs in merge.items():
            self.y[self.y[self.config["y_label"]].isin(labs)] = cls
            self.y_test[self.y_test[self.config["y_label"]].isin(labs)] = cls

        self.encode_labels()

    def encode_labels(self) -> None:
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y)
        self.label_encoder_ohe = OneHotEncoder(sparse_output=False)
        self.label_encoder_ohe.fit(self.y)

    def decode_labels(self):
        pass

    def num_classes(self) -> int:
        return self.y[self.config["y_label"]].nunique()

    def num_features(self) -> int:
        return len(self.X.columns)

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

    def memory_usage_mb(self, df) -> float:
        return df.memory_usage().sum() / 1024**2

    def reduce_mem_df(self, df) -> None:
        for col in df.columns:
            col_type = self.X[col].dtype

            if col_type != object:
                c_min = self.X[col].min()
                c_max = self.X[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
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

    def reduce_mem(self) -> None:
        """iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = self.memory_usage_mb(self.X) + self.memory_usage_mb(self.X_test)
        console.print("💾 Memory usage of dataframe is {:.2f} MB...".format(start_mem))

        with console.status(
            "Reducing memory usage...", spinner=SPINNER, refresh_per_second=REFRESH
        ) as status:
            self.reduce_mem_df(self.X)
            self.reduce_mem_df(self.X_test)

        end_mem = self.memory_usage_mb(self.X) + self.memory_usage_mb(self.X_test)
        console.print(
            "💾 Memory usage after optimization: {:.2f} MB...".format(end_mem)
        )
        console.print(
            "✅ Memory usage reduced by {:.1f}%...".format(
                100 * (start_mem - end_mem) / start_mem
            )
        )

    def reduce_mem_gen(self) -> None:
        """iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """
        start_mem = self.memory_usage_mb(self.X_gen)
        console.print("💾 Memory usage of dataframe is {:.2f} MB...".format(start_mem))

        with console.status(
            "Reducing memory usage...", spinner=SPINNER, refresh_per_second=REFRESH
        ) as status:
            self.reduce_mem_df(self.X_gen)

        end_mem = self.memory_usage_mb(self.X_gen)
        console.print(
            "💾 Memory usage after optimization: {:.2f} MB...".format(end_mem)
        )
        console.print(
            "✅ Memory usage reduced by {:.1f}%...".format(
                100 * (start_mem - end_mem) / start_mem
            )
        )
