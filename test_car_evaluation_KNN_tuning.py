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
from synthtab.evaluators import KNN, KNN, XGBoost, MLP
from synthtab.tuners import (
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
from synthtab.data import Config, Dataset
from synthtab.utils import console

config = Config("configs/car_evaluation_knn.json")

dataset = Dataset(config)

trials = 10

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ROS(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = SMOTE(dataset)
# evaluator = KNN(generator)
# tuner = SMOTETuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ADASYN(dataset, sampling_strategy="minority")
# evaluator = KNN(generator)
# tuner = ADASYNTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = TVAE(dataset)
# evaluator = KNN(generator)
# tuner = TVAETuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTGAN(dataset)
# evaluator = KNN(generator)
# tuner = CTGANTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = GaussianCopula(dataset)
# evaluator = KNN(generator)
# tuner = GaussianCopulaTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CopulaGAN(dataset)
# evaluator = KNN(generator)
# tuner = CopulaGANTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGAN(
#     dataset,
#     test_ratio=0.10,
# )
# evaluator = KNN(generator)
# tuner = CTABGANTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGANPlus(
#     dataset,
#     test_ratio=0.10,
# )
# evaluator = KNN(generator)
# tuner = CTABGANPlusTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = AutoDiffusion(dataset)
# evaluator = KNN(generator)
# tuner = AutoDiffusionTuner(evaluator, trials)
# tuner.tune()
# generator = tuner.generator
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = ForestDiffusion(dataset)
evaluator = KNN(generator)
tuner = ForestDiffusionTuner(evaluator, trials)
tuner.tune()
generator = tuner.generator
generator.generate()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = GReaT(dataset)
evaluator = KNN(generator)
tuner = GReaTTuner(evaluator, trials)
tuner.tune()
generator = tuner.generator
generator.generate()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(dataset)
evaluator = KNN(generator)
tuner = TabulaTuner(evaluator, trials)
tuner.tune()
generator = tuner.generator
generator.generate()
generator.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
