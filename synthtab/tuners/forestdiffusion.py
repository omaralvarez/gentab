from . import Tuner
from synthtab.generators import ForestDiffusion
from synthtab.evaluators import Evaluator
from synthtab.utils import console, SPINNER, REFRESH

import optuna


class ForestDiffusionTuner(Tuner):
    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(evaluator)

    def objective(self, trial: optuna.trial.Trial) -> float:
        n_t = trial.suggest_int("n_t", 20, 60)
        model = trial.suggest_categorical(
            "model", ["xgboost", "random_forest", "lgbm", "catboost"]
        )
        diffusion_type = trial.suggest_categorical("diffusion_type", ["flow", "vp"])
        max_depth = trial.suggest_int("max_depth", 3, 9, step=2)
        n_estimators = trial.suggest_int("n_estimators", 2, 100)
        eta = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        tree_method = trial.suggest_categorical(
            "tree_method", ["exact", "approx", "hist"]
        )
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
        subsample = trial.suggest_float("subsample", 0.2, 1.0)
        num_leaves = trial.suggest_int("num_leaves", 2, 256)
        duplicate_K = trial.suggest_int("duplicate_K", 2, 50)

        eps = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
        beta_min = trial.suggest_float("beta_min", 1e-8, 0.5, log=True)
        beta_max = trial.suggest_int("beta_max", 2, 10)
        batch_size = trial.suggest_int("batch_size", 512, 16384)

        # TODO Maybe add batch sizes
        self.generator = ForestDiffusion(
            self.dataset,
            n_t=n_t,
            model=model,
            diffusion_type=diffusion_type,
            max_depth=max_depth,
            n_estimators=n_estimators,
            eta=eta,
            tree_method=tree_method,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            num_leaves=num_leaves,
            duplicate_K=duplicate_K,
            eps=eps,
            beta_min=beta_min,
            beta_max=beta_max,
            n_batch=batch_size,
        )
        self.generator.generate()

        trial.set_user_attr("generator", self.generator)

        acc, mcc = self.evaluator.evaluate()

        return mcc
