import json
import os

from torch.utils.data import Dataset
import torch

from vi_sample import create_examples
from vi_preprocessing import create_inputs_targets

class AcrDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.X, self.Y = create_inputs_targets(self.examples)
            
    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.X[0][idx], dtype=torch.int64)
        input_type_ids = torch.tensor(self.X[1][idx], dtype=torch.int64)
        attention_mask = torch.tensor(self.X[2][idx], dtype=torch.float)
        start_token_idx = torch.tensor(self.Y[0][idx], dtype=torch.int64)
        end_token_idx = torch.tensor(self.Y[1][idx], dtype=torch.int64)
        label = torch.tensor(self.Y[2][idx], dtype=torch.int8)
        return (input_ids, input_type_ids, attention_mask, start_token_idx, end_token_idx, label)
