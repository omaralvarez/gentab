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
)
from synthtab.data.config import Config
from synthtab.data.dataset import Dataset
from synthtab.console import console

config = Config("datasets/playnet/info.json")

dataset = Dataset(config)
dataset.reduce_size(
    {
        "left_attack": 0.65,
        "right_attack": 0.65,
        "right_transition": 0.65,
        "left_transition": 0.65,
        "time_out": 0.65,
    }
)
dataset.reduce_mem()

console.print(dataset.class_counts(), dataset.row_count())
generator = ROS(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = SMOTE(dataset)
generator.generate({"right_transition": 83, "time_out": 153})
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = ADASYN(dataset)
generator.generate({"right_transition": 83, "time_out": 153})
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = TVAE(dataset)
generator.generate()
dataset.save_to_disk(generator)
console.print(dataset.generated_class_counts(), dataset.generated_row_count())

console.print(dataset.class_counts(), dataset.row_count())
generator = CTGAN(dataset)
generator.generate({"right_transition": 83, "time_out": 153})
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
    categorical_columns=[dataset.config["y_label"]],
    mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
    problem_type={"Classification": dataset.config["y_label"]},
    epochs=10,
)
generator.generate({"right_transition": 83, "time_out": 153})
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)

console.print(dataset.class_counts(), dataset.row_count())
generator = CTABGANPlus(
    dataset,
    test_ratio=0.10,
    categorical_columns=[dataset.config["y_label"]],
    mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
    problem_type={"Classification": dataset.config["y_label"]},
    epochs=10,
)
generator.generate()
console.print(dataset.generated_class_counts(), dataset.generated_row_count())
dataset.save_to_disk(generator)
