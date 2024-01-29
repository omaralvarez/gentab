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
from synthtab.utils import console

config = Config("datasets/congressional/info.json")

dataset = Dataset(config)
# dataset.reduce_mem()

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ROS(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = SMOTE(dataset)
# generator.generate()
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# # console.print(dataset.class_counts(), dataset.row_count())
# # generator = ADASYN(dataset, sampling_strategy="minority")
# # # generator.generate({"right_transition": 83, "time_out": 153})
# # generator.generate()
# # dataset.save_to_disk(generator)
# # console.print(dataset.generated_class_counts(), dataset.generated_row_count())

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

cat = list(dataset.X.columns)
cat.append(dataset.config["y_label"])
# console.print(dataset.class_counts(), dataset.row_count())
console.print(cat)
# generator = CTABGAN(
#     dataset,
#     test_ratio=0.1,
#     categorical_columns=cat,
#     problem_type={"Classification": dataset.config["y_label"]},
#     epochs=1000,
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGANPlus(
#     dataset,
#     test_ratio=0.05,
#     epochs=2000,
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

# console.print(dataset.class_counts(), dataset.row_count())
# generator = GReaT(
#     dataset,
#     epochs=1000,
#     max_length=1024,
#     temperature=0.6,
#     batch_size=32,
#     max_tries_per_batch=4096,
#     n_samples=8192,
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = Tabula(
#     dataset,
#     # categorical_columns=[dataset.config["y_label"]],
#     epochs=1000,
#     max_length=1024,
#     temperature=0.6,
#     batch_size=32,
#     max_tries_per_batch=4096,
#     n_samples=8192,
# )
# generator.generate()
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)
