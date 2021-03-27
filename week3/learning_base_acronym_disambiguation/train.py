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

tokenizer = BertWordPieceTokenizer("slow_token/vocab.txt", lowercase=True)

with open("../AAAI-21-SDU-shared-task-2-AD/dataset/diction.json", "r") as f:
    diction = json.load(f)
with open("pos_data/train_pos_data.json", "r", encoding="UTF-8") as f:
    pos_train = json.load(f)
    
with open("neg_data/train_neg_data.json", "r", encoding="UTF-8") as f:
    neg_train = json.load(f)

examples_pos = create_examples(pos_train, "Create pos training points", tokenizer)

examples = create_examples(neg_train, "Create neg training points", tokenizer)

examples.extend(examples_pos)

X, Y = create_inputs_targets(examples)

train_data = TensorDataset(torch.tensor(X[0], dtype=torch.int64),  #input_ids
                            torch.tensor(X[1], dtype=torch.int64), #input_type_ids
                            torch.tensor(X[2], dtype=torch.float), #attention_mask
                            torch.tensor(Y[0], dtype=torch.int64), #start_token_acr
                            torch.tensor(Y[1], dtype=torch.int64), #end_token_acr
                            torch.tensor(Y[2], dtype=torch.float)) #label

train_sampler = RandomSampler(train_data)
train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AcrBertModel.from_pretrained("bert-base-uncased").to(device=device)
model.save_pretrained("bert-base")
print(model)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = torch.optim.Adam(lr=1e-5, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)

loss_fn = nn.BCELoss()

loss_fn = nn.BCELoss()

for epoch in range(1, 10):
    print("Training epoch ", str(epoch))
    training_pbar = tqdm(total=len(train_data),
                         position=0, leave=True,
                         file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
    model.train()
    tr_loss = 0
    
    
    for step, batch in enumerate(train_data_loader):
        batch = tuple(t.to(device) for t in batch)
        input_word_ids, input_type_ids, input_mask, start_token_idx, end_token_idx, label= batch
        optimizer.zero_grad()
        output, _ = model(input_ids=input_word_ids,
                        token_type_ids=input_type_ids,
                        attention_mask=input_mask,
                        start_token_idx=start_token_idx,
                        end_token_idx=end_token_idx)
        # print(loss)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        if step % 10000 == 0:
          print(f"Loss = {loss} / step {step}")
        training_pbar.update(input_word_ids.size(0))
    training_pbar.close()
    print(f"\n Binary Cross Entropy loss = {tr_loss:.8f}/ epoch {epoch}")
    torch.save(model.state_dict(), "./weights_" + str(epoch) + ".pth")