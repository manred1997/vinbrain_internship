import json
import random
import sys

from tqdm import tqdm
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Sample import Sample, create_examples
from preprocessing import create_inputs_targets

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import BertWordPieceTokenizer

from model import AcrBertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertWordPieceTokenizer("slow_token/vocab.txt", lowercase=True)

with open("dev_data/dev_data.json", "r") as f:
    dev = json.load(f)

model = Ả