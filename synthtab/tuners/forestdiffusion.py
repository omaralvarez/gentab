from . import Tuner
from synthtab.generators import ForestDiffusion
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class ForestDiffusionTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        trials: int,
        *args,
        min_batch: int = 512,
        max_batch: int = 16384,
        max_tries_per_batch: int = 8192,
        timeout: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            evaluator,
            trials,
            min_batch=min_batch,
            max_batch=max_batch,
            max_tries_per_batch=max_tries_per_batch,
            timeout=timeout,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        # batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        n_t = trial.suggest_int("n_t", 20, 80)
        model = trial.suggest_categorical(
            "model", ["xgboost", "random_forest", "lgbm", "catboost"]
        )
        diffusion_type = trial.suggest_categorical("diffusion_type", ["flow", "vp"])
        max_depth = trial.suggest_int("max_depth", 3, 9)
        n_estimators = trial.suggest_int("n_estimators", 20, 200, step=2)
        eta = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # tree_method = trial.suggest_categorical(
        #     "tree_method", ["exact", "approx", "hist"]
        # )
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
        subsample = trial.suggest_float("subsample", 0.2, 1.0)
        num_leaves = trial.suggest_int("num_leaves", 2, 256)
        duplicate_K = trial.suggest_int("duplicate_K", 2, 50)

        eps = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
        beta_min = trial.suggest_float("beta_min", 1e-8, 0.5, log=True)
        beta_max = trial.suggest_categorical("beta_max", [2, 4, 8, 16])

        self.generator = ForestDiffusion(
            self.dataset,
            n_t=n_t,
            model=model,
            diffusion_type=diffusion_type,
            max_depth=max_depth,
            n_estimators=n_estimators,
            eta=eta,
            # tree_method=tree_method,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            num_leaves=num_leaves,
            duplicate_K=duplicate_K,
            eps=eps,
            beta_min=beta_min,
            beta_max=beta_max,
            # n_batch=batch_size,
            max_tries_per_batch=self.max_tries_per_batch,
        )
        self.generator.generate()

        trial.set_user_attr("timing", self.generator.timer.history)
        trial.set_user_attr("dataset", self.dataset)

        acc, mcc = self.evaluator.evaluate(validation=True)

        return mcc
