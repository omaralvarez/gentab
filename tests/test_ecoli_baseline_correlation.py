from gentab.generators import (
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

config = Config("configs/ecoli_cr.json")

dataset = Dataset(config)

n_samples = dataset.class_counts().to_dict()

console.print(dataset.class_counts(), dataset.row_count())
generator = TVAE(dataset)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTGAN(dataset)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GaussianCopula(dataset)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CopulaGAN(dataset)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGAN(
    dataset,
    test_ratio=0.1,
    epochs=700,
)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGANPlus(
    dataset,
    test_ratio=0.1,
    high_quality=False,
    epochs=1000,
)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = AutoDiffusion(dataset, batch_size=64)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ForestDiffusion(dataset, n_jobs=1, duplicate_K=4, n_estimators=100)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = GReaT(
    dataset,
    epochs=600,
    max_length=1024,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = Tabula(
    dataset,
    epochs=600,
    max_length=1024,
    temperature=0.6,
    batch_size=32,
    max_tries_per_batch=4096,
    n_samples=8192,
)
generator.generate(n_samples=n_samples, append=False)
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
