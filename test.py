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
console.print(dataset.y.value_counts())
dataset.reduce_size({"left_attack": 0.5, "right_attack": 0.5})
console.print(dataset.y.value_counts())
dataset.reduce_mem()

# generator = ROS(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = SMOTE(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = ADASYN(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = TVAE(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = CTGAN(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = GaussianCopula(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

# generator = CopulaGAN(dataset)
# generator.generate()
# dataset.save_to_disk(generator)

generator = CTABGAN(
    dataset,
    test_ratio=0.10,
    categorical_columns=[dataset.config["y_label"]],
    mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
    problem_type={"Classification": dataset.config["y_label"]},
    epochs=10,
)
generator.generate()
dataset.save_to_disk(generator)

generator = CTABGANPlus(
    dataset,
    test_ratio=0.10,
    categorical_columns=[dataset.config["y_label"]],
    mixed_columns=dict([(c, [0.0]) for c in dataset.X.columns]),
    problem_type={"Classification": dataset.config["y_label"]},
    epochs=10,
)
generator.generate()
dataset.save_to_disk(generator)
