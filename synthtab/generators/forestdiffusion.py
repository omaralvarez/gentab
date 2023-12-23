from . import Generator
from synthtab.utils import console, SPINNER, REFRESH

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
        duplicate_K=100,  # number of different noise sample per real data sample
        bin_indexes=[],  # vector which indicates which column is binary
        cat_indexes=[],  # vector which indicates which column is categorical (>=3 categories)
        int_indexes=[],  # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
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
        self.__name__ = "ForestDiffusion"
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
        self.bin_indexes = bin_indexes
        self.cat_indexes = cat_indexes
        self.int_indexes = int_indexes
        self.true_min_max_values = true_min_max_values
        self.gpu_hist = gpu_hist
        self.eps = eps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_jobs = n_jobs

    def preprocess(self) -> None:
        if self.dataset.config["task_type"] in ["binary", "multiclass"]:
            self.X = self.dataset.X.to_numpy()
            self.codes = pd.Categorical(self.dataset.y[self.dataset.config["y_label"]])
            self.y = self.codes.codes
            self.Xy = np.concatenate((self.X, np.expand_dims(self.y, axis=1)), axis=1)
        else:
            # TODO We need to deal with all categorical indexes in the pandas df
            # set the indexes in the config for everything and then have them available
            # for the methods that need them
            self.Xy = self.dataset.get_single_df().to_numpy()

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
            )

    def sample(self) -> pd.DataFrame:
        Xy_gen = self.forest_model.generate(batch_size=self.batch_size)
        if self.dataset.config["task_type"] in ["binary", "multiclass"]:
            df = pd.DataFrame(data=Xy_gen[:, :-1], columns=self.dataset.X.columns)
            df[self.dataset.config["y_label"]] = pd.DataFrame({"tmp": Xy_gen[:, -1]})[
                "tmp"
            ].apply(lambda x: self.codes.categories[int(x)])
        else:
            df = pd.DataFrame(data=Xy_gen, columns=self.dataset.get_single_df().columns)

        return df
