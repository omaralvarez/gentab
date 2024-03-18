import numpy as np
import random
import torch
import os

SEED = 42

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
