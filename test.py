from synthtab.generators import ROS,SMOTE,ADASYN,TVAE,CTGAN,GaussianCopula,CopulaGAN,CTABGAN 
from synthtab.progress import ProgressBar
from synthtab.algorithm import Algorithm
from synthtab.data.config import Config
from synthtab.data.dataset import Dataset
from synthtab.console import console

config = Config('datasets/playnet/info.json')

dataset = Dataset(config)
# remember to deactivate if not working sampling and keeps on having weird things
dataset.reduce_mem()

generator = ROS(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = SMOTE(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = ADASYN(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = TVAE(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = CTGAN(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = GaussianCopula(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = CopulaGAN(dataset)
generator.generate()
dataset.save_to_disk(str(generator))

generator = CTABGAN(dataset)
generator.generate()
dataset.save_to_disk(str(generator))
