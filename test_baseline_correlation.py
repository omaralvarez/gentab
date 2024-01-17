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

config = Config("configs/playnet_cr.json")

dataset = Dataset(config)
console.print(dataset.class_counts(), dataset.row_count())
dataset.reduce_size({
    "left_attack": 0.97,
    "right_attack": 0.97,
    "right_transition": 0.9,
    "left_transition": 0.9,
    "time_out": 0.8,
    "left_penal": 0.5,
    "right_penal": 0.5,
})
dataset.merge_classes({
    "attack": ["left_attack", "right_attack"],
    "transition": ["left_transition", "right_transition"],
    "penalty": ["left_penal", "right_penal"],
})
console.print(dataset.class_counts(), dataset.row_count())
dataset.reduce_mem()

n_samples = dataset.class_counts().to_dict()

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ROS(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = SMOTE(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = ADASYN(dataset, sampling_strategy="minority")
# # generator.generate({"right_transition": 83, "time_out": 153})
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = TVAE(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTGAN(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = GaussianCopula(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CopulaGAN(dataset)
# generator.generate(n_samples=n_samples, append=False)
# dataset.save_to_disk(generator)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGAN(
#     dataset,
#     test_ratio=0.10,
# )
# generator.generate(n_samples=n_samples, append=False)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

# console.print(dataset.class_counts(), dataset.row_count())
# generator = CTABGANPlus(
#     dataset,
#     test_ratio=0.10,
# )
# generator.generate(n_samples=n_samples, append=False)
# console.print(dataset.generated_class_counts(), dataset.generated_row_count())
# dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = AutoDiffusion(dataset)
generator.generate(n_samples=n_samples, append=False)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
generator.generate(n_samples=n_samples, append=False)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = GReaT(
    dataset,
    epochs=15,
    max_length=2000,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
generator.generate(n_samples=n_samples, append=False)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(
    dataset,
    # categorical_columns=[dataset.config["y_label"]],
    epochs=15,
    max_length=1024,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
generator.generate(n_samples=n_samples, append=False)
generator.save_to_disk()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
