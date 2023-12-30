from synthtab.generators import (
    ROS,
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
from synthtab.evaluators import LightGBM
from synthtab.tuners import (
    SMOTETuner,
    ADASYNTuner,
    TVAETuner,
    CTGANTuner,
    GaussianCopulaTuner,
    CopulaGANTuner,
    # CTABGANTuner,
    # CTABGANPlusTuner,
    # AutoDiffusionTuner,
    # ForestDiffusionTuner,
    # Tabula,
    # GReaT,
)
from synthtab.data import Config, Dataset
from synthtab.utils import console

config = Config("datasets/playnet/info.json")

dataset = Dataset(config)
dataset.reduce_size({
    "left_attack": 0.9,
    "right_attack": 0.9,
    "right_transition": 0.9,
    "left_transition": 0.9,
    "time_out": 0.9,
    "left_penalty": 0.1,
    "right_penalty": 0.25,
})
dataset.reduce_mem()

trials = 10

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ROS(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = SMOTE(dataset)
evaluator = LightGBM(generator)
tuner = SMOTETuner(evaluator, trials)
tuner.tune()
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ADASYN(dataset, sampling_strategy="minority")
evaluator = LightGBM(generator)
tuner = ADASYNTuner(evaluator, trials)
tuner.tune()
# generator.generate({"right_transition": 83, "time_out": 153})
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = TVAE(dataset)
evaluator = LightGBM(generator)
tuner = TVAETuner(evaluator, trials)
tuner.tune()
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTGAN(dataset)
evaluator = LightGBM(generator)
tuner = CTGANTuner(evaluator, trials)
tuner.tune()
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GaussianCopula(dataset)
evaluator = LightGBM(generator)
tuner = GaussianCopulaTuner(evaluator, trials)
tuner.tune()
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CopulaGAN(dataset)
evaluator = LightGBM(generator)
tuner = CopulaGANTuner(evaluator, trials)
tuner.tune()
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGAN(
#     dataset,
#     test_ratio=0.10,
# )
# evaluator = LightGBM(generator)
# tuner = CTABGANTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGANPlus(
#     dataset,
#     test_ratio=0.10,
# )
# evaluator = LightGBM(generator)
# tuner = CTABGANPlusTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = AutoDiffusion(dataset)
# evaluator = LightGBM(generator)
# tuner = AutoDiffusionTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
# evaluator = LightGBM(generator)
# tuner = ForestDiffusionTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = GReaT(
#     dataset,
#     epochs=50,
#     max_length=2000,
#     temperature=0.6,
#     batch_size=32,
#     max_tries_per_batch=4096,
#     n_samples=8192,
# )
# evaluator = LightGBM(generator)
# tuner = GReaTTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = Tabula(
#     dataset,
#     # categorical_columns=[dataset.config["y_label"]],
#     epochs=50,
#     max_length=1024,
#     temperature=0.6,
#     batch_size=32,
#     max_tries_per_batch=4096,
#     n_samples=8192,
# )
# evaluator = LightGBM(generator)
# tuner = TabulaTuner(evaluator, trials)
# tuner.tune()
# generator.generate()
# generator.save_to_disk()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
