from synthtab.generators import ROS,SMOTE,ADASYN,TVAE,CTGAN,GaussianCopula,CopulaGAN 
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

generator = SMOTE(dataset)
generator.generate()

generator = ADASYN(dataset)
generator.generate()

generator = TVAE(dataset)
generator.generate(1000)

generator = CTGAN(dataset)
generator.generate(1000)

generator = GaussianCopula(dataset)
generator.generate(1000)

generator = CopulaGAN(dataset)
generator.generate(1000)
dataset.save_to_disk()
