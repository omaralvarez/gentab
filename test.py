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
from synthtab.data import Config, Dataset
from synthtab.console import console

from sklearn import svm

config = Config("datasets/playnet/info.json")

dataset = Dataset(config)
dataset.reduce_size({
    "left_attack": 0.65,
    "right_attack": 0.65,
    "right_transition": 0.65,
    "left_transition": 0.65,
    "time_out": 0.65,
})
dataset.reduce_mem()

eval_model = svm.SVC(gamma=0.001, C=100.0)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ROS(dataset)
# generator.generate()
# generator.evaluate(eval_model)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = SMOTE(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ADASYN(dataset)
# # generator.generate({"right_transition": 83, "time_out": 153})
# generator.generate()
# dataset.save_to_disk(generator, sampling_strategy="minority")
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = TVAE(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTGAN(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = GaussianCopula(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CopulaGAN(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGAN(
#     dataset,
#     test_ratio=0.10,
#     categorical_columns=[dataset.config["y_label"]],
#     mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
#     problem_type={"Classification": dataset.config["y_label"]},
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGANPlus(
#     dataset,
#     test_ratio=0.10,
#     categorical_columns=[dataset.config["y_label"]],
#     # TODO Abstract this.
#     mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
#     problem_type={"Classification": dataset.config["y_label"]},
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = AutoDiffusion(dataset)
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(
    dataset,
    # categorical_columns=[dataset.config["y_label"]],
    epochs=100,
    max_length=2000,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=262144,
    trained_model="model_playnet.pt",
)
generator.generate()
generator.evaluate(eval_model)
generator.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())


# console.print(dataset.class_counts(), dataset.row_count())
# generator = GReaT(
#     dataset,
#     epochs=10,
#     max_length=2000,
#     temperature=0.6,
#     batch_size=32,
#     max_tries_per_batch=4096,
#     n_samples=8192,
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# TODO Timing..
