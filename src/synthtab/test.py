from generator import Generator
from progress import ProgressBar
from algorithm import Algorithm
from config import Config
from dataset import Dataset

from rich import print
import json

config = Config('data/playnet/info.json')
dataset = Dataset(config)
dataset.reduce_mem()

generator = Generator(Algorithm(config, dataset), dataset)

print(generator, 'PlayNet')

# with ProgressBar().progress as p:
#     # for i in track(range(20), description="Processing..."):
#     #     time.sleep(1)  # Simulate work being done
#     for n in p.track(range(1000)):
#         n = n - 2
#         total = n + total

# print(total)
