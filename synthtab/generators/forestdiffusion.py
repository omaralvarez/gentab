from . import Generator
from synthtab.utils import console

from ForestDiffusion import ForestDiffusionModel
import pandas as pd
import os
import numpy as np

os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"


class ForestDiffusion(Generator):
    def __init__(
        self,
        dataset,
        n_t=50,  # number of noise level
        model="xgboost",  # xgboost, random_forest, lgbm, catboost
        diffusion_type="flow",  # vp, flow (flow is better, but only vp can be used for imputation)
        max_depth=7,
        n_estimators=100,
        eta=0.3,  # xgboost hyperparameters
        tree_method="hist",
        reg_lambda=0.0,
        reg_alpha=0.0,
        subsample=1.0,  # xgboost hyperparameters
        num_leaves=31,  # lgbm hyperparameters
        duplicate_K=4,  # number of different noise sample per real data sample
        true_min_max_values=None,  # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
        gpu_hist=False,  # using GPU or not with xgboost
        eps=1e-3,
        beta_min=0.1,
        beta_max=8,
        n_jobs=-1,  # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
        batch_size=8192,
        max_tries_per_batch=4096,
        **xgboost_kwargs,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        self.n_t = n_t
        self.model = model
        self.diffusion_type = diffusion_type
        self.max_depth = max_depth
        self.tree_method = tree_method
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.duplicate_K = duplicate_K
        self.true_min_max_values = true_min_max_values
        self.gpu_hist = gpu_hist
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_jobs = n_jobs
        # if >0 use the data iterator with the specified number of batches
        self.n_batch = batch_size

    def preprocess(self) -> None:
        X = self.dataset.encode_categories(self.dataset.X)

        # due to forestdiffusion not using pd.DF we need to get col idx...
        # vector which indicates which column is binary
        self.bin_indexes = [X.columns.get_loc(k) for k in self.config["binary_columns"]]
        # vector which indicates which column is categorical (>=3 categories)
        self.cat_indexes = [
            X.columns.get_loc(k) for k in self.config["categorical_columns"]
        ]
        # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
        self.int_indexes = [
            X.columns.get_loc(k) for k in self.config["integer_columns"]
        ]

        self.X = X.to_numpy()

        if self.dataset.config["task_type"] in ["binary", "multiclass"]:
            self.codes = pd.Categorical(self.dataset.y[self.dataset.config["y_label"]])
            self.y = self.codes.codes

        self.Xy = np.concatenate((self.X, np.expand_dims(self.y, axis=1)), axis=1)

    def train(self) -> None:
        if self.dataset.config["task_type"] in ["binary", "multiclass"]:
            self.forest_model = ForestDiffusionModel(
                self.X,
                label_y=self.y,
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                tree_method=self.tree_method,
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                eta=self.eta,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                subsample=self.subsample,
                duplicate_K=self.duplicate_K,
                bin_indexes=self.bin_indexes,
                cat_indexes=self.cat_indexes,
                int_indexes=self.int_indexes,
                true_min_max_values=self.true_min_max_values,
                gpu_hist=self.gpu_hist,
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                seed=self.seed,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
            )
        else:
            self.forest_model = ForestDiffusionModel(
                self.Xy,
                n_t=self.n_t,
                model=self.model,
                diffusion_type=self.diffusion_type,
                max_depth=self.max_depth,
                tree_method=self.tree_method,
                num_leaves=self.num_leaves,
                n_estimators=self.n_estimators,
                eta=self.eta,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                subsample=self.subsample,
                duplicate_K=self.duplicate_K,
                bin_indexes=self.bin_indexes,
                cat_indexes=self.cat_indexes,
                int_indexes=self.int_indexes,
                true_min_max_values=self.true_min_max_values,
                gpu_hist=self.gpu_hist,
                eps=self.eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                seed=self.seed,
                n_jobs=self.n_jobs,
                n_batch=self.n_batch,
            )

    def sample(self) -> pd.DataFrame:
        Xy_gen = self.forest_model.generate(batch_size=self.batch_size)

        df = pd.DataFrame(data=Xy_gen[:, :-1], columns=self.dataset.X.columns)

        # This is due to the library not converting back to integer binary
        # and categorical variables and not converting back to strings on its own
        int_cls = self.config["binary_columns"] + self.config["categorical_columns"]
        df[int_cls] = df[int_cls].apply(lambda col: col.round().astype("int64"))
        df = self.dataset.decode_categories(df)

        # TODO Use decode labels in dataset
        if self.dataset.config["task_type"] in ["binary", "multiclass"]:
            df[self.dataset.config["y_label"]] = pd.DataFrame({"tmp": Xy_gen[:, -1]})[
                "tmp"
            ].apply(lambda x: self.codes.categories[int(x)])
        else:
            df[self.dataset.config["y_label"]] = Xy_gen[:, -1]

        return df
