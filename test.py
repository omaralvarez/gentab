from synthtab.generators import ROS,SMOTE,ADASYN,TVAE,CTGAN,GaussianCopula,CopulaGAN,CTABGAN,CTABGANPlus 
from synthtab.data.config import Config
from synthtab.data.dataset import Dataset
from synthtab.console import console

config = Config('datasets/playnet/info.json')

dataset = Dataset(config)
# remember to deactivate if not working sampling and keeps on having weird things
dataset.reduce_mem()

# generator = ROS(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = SMOTE(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = ADASYN(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = TVAE(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = CTGAN(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = GaussianCopula(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

# generator = CopulaGAN(dataset)
# generator.generate()
# dataset.save_to_disk(str(generator))

generator = CTABGAN(
    dataset, 
    test_ratio=0.10,
    categorical_columns=[dataset.config['y_label']], 
    mixed_columns=dict([(c,[0.0]) for c in dataset.X.columns]),
    problem_type={'Classification': '#play'},
    epochs=10
)
generator.generate()
dataset.save_to_disk(str(generator))

generator = CTABGANPlus(
    dataset, 
    test_ratio=0.10,
    categorical_columns=[dataset.config['y_label']], 
    mixed_columns=dict([(c,[0.0]) for c in dataset.X.columns]),
    problem_type={'Classification': '#play'},
    epochs=10
)
generator.generate()
dataset.save_to_disk(str(generator))
