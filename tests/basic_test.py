import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir,os.path.pardir)))
from megatron.data import indexed_dataset

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')
SEED = 0


def main():
    pass


if __name__ == '__main__':
    main()
