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
from gentab.data import Config, Dataset
from gentab.utils import console

config = Config("configs/mushroom.json")

dataset = Dataset(config)
dataset.reduce_size({"e": 0.0, "p": 0.6})

console.print(dataset.class_counts(), dataset.row_count())
generator = TVAE(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTGAN(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GaussianCopula(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CopulaGAN(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGAN(
    dataset,
    test_ratio=0.10,
)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGANPlus(
    dataset,
    test_ratio=0.10,
)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = AutoDiffusion(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

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
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(
    dataset,
    epochs=15,
    max_length=1024,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = SMOTE(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ADASYN(dataset, sampling_strategy="minority")
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
