from gentab.generators import (
    SMOTE,
    ADASYN,
    TVAE,
    CTGAN,
    GaussianCopula,
    CopulaGAN,
    CTABGAN,
    CTABGANPlus,
    AutoDiffusion,
    ForestDiffusion,
    Tabula,
    GReaT,
)
from gentab.evaluators import CatBoost
from gentab.tuners import (
    SMOTETuner,
    ADASYNTuner,
    TVAETuner,
    CTGANTuner,
    GaussianCopulaTuner,
    CopulaGANTuner,
    CTABGANTuner,
    CTABGANPlusTuner,
    AutoDiffusionTuner,
    ForestDiffusionTuner,
    TabulaTuner,
    GReaTTuner,
)
from gentab.data import Config, Dataset
from gentab.utils import console

config = Config("configs/oil.json")

dataset = Dataset(config)

trials = 10

console.print(dataset.class_counts(), dataset.row_count())
generator = TVAE(dataset)
evaluator = CatBoost(generator)
tuner = TVAETuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTGAN(dataset)
evaluator = CatBoost(generator)
tuner = CTGANTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GaussianCopula(dataset)
evaluator = CatBoost(generator)
tuner = GaussianCopulaTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CopulaGAN(dataset)
evaluator = CatBoost(generator)
tuner = CopulaGANTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGAN(
    dataset,
    test_ratio=0.10,
)
evaluator = CatBoost(generator)
tuner = CTABGANTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGANPlus(
    dataset,
    high_quality=False,
    test_ratio=0.10,
)
evaluator = CatBoost(generator)
tuner = CTABGANPlusTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = AutoDiffusion(dataset)
evaluator = CatBoost(generator)
tuner = AutoDiffusionTuner(
    evaluator, trials, min_batch=128, max_batch=512, min_epochs=500, max_epochs=8000
)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
evaluator = CatBoost(generator)
tuner = ForestDiffusionTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GReaT(
    dataset,
    epochs=100,
    max_length=1024,
    temperature=0.4,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
evaluator = CatBoost(generator)
tuner = GReaTTuner(evaluator, trials, min_epochs=800, max_epochs=1000)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(
    dataset,
    epochs=100,
    max_length=1024,
    temperature=0.4,
    batch_size=32,
    max_tries_per_batch=16384,
    n_samples=8192,
)
evaluator = CatBoost(generator)
tuner = TabulaTuner(
    evaluator, trials, min_epochs=500, max_epochs=1000, max_tries_per_batch=16384
)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = SMOTE(dataset)
evaluator = CatBoost(generator)
tuner = SMOTETuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ADASYN(dataset, sampling_strategy="minority")
evaluator = CatBoost(generator)
tuner = ADASYNTuner(evaluator, trials)
tuner.tune()
tuner.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
