import json
import random
import sys
import argparse

from tqdm import tqdm
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tokenizers import BertWordPieceTokenizer

from vi_model_bert import AcrBertModel
from vi_sample import create_examples
from vi_split_train_dev import split_train_dev
from vi_dataset import AcrDataset
# from vi_preprocessing import create_inputs_targets

tokenizer = BertWordPieceTokenizer("slow_token/vocab.txt", lowercase=True)

def train(model, optimizer, train_data_loader, epoch, loss_fn, device):
    # print("Training epoch ", str(epoch+1))
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    
    with tqdm.trange(len(train_data_loader), desc=f'Training dataset for epoch {epoch + 1}...') as t:
        for step, batch in enumerate(train_data_loader):
            batch = tuple(t.to(device) for t in batch)
            input_word_ids, input_type_ids, input_mask, start_token_idx, end_token_idx, label= batch
            optimizer.zero_grad()
            outputs, _ = model(input_ids=input_word_ids,
                            token_type_ids=input_type_ids,
                            attention_mask=input_mask,
                            start_token_idx=start_token_idx,
                            end_token_idx=end_token_idx)
            # print(loss)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            if step % 1000 == 0:
                print(f"Loss = {loss} / step {step}")
            t.set_postfix(loss=f'{loss:05.5f}')
            t.update()
    print(f"\n Binary Cross Entropy loss = {tr_loss/nb_tr_steps:.8f}/ epoch {epoch+1} on training set")
    torch.save(model.state_dict(), "./weights_" + str(epoch+1) + ".pth")
    return tr_loss

def test(model, dev_data_loader, device, loss_fn):
    correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dev_data_loader):
            batch = tuple(t.to(device) for t in batch)
            input_word_ids, input_type_ids, input_mask, start_token_idx, end_token_idx, label= batch

            outputs, _ = model(input_ids=input_word_ids,
                            token_type_ids=input_type_ids,
                            attention_mask=input_mask,
                            start_token_idx=start_token_idx,
                            end_token_idx=end_token_idx)
            loss = loss_fn(outputs, label)
            tr_loss += loss.item()
            nb_tr_steps += 1
            preds = (outputs > 0.5).type(torch.int8)
            correct += sum(preds == label).item()
            

    print(f"\n Binary Cross Entropy loss = {tr_loss/nb_tr_steps:.8f}/ epoch {epoch} on dev set")
    return correct, tr_loss

def main(model, raw_data, writer, params, device, optimizer, loss_fn):

    trainset, devset = split_train_dev(raw_data, test_size=0.1)
    examples_train = create_examples(trainset, "Create exmaples for training set", tokenizer)
    examples_dev = create_examples(devset, "Create examples for dev set", tokenizer)

    train_data = AcrDataset(examples_train)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, batch_size=params.batch_size_train, sampler=train_sampler)

    dev_data = AcrDataset(examples_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_data_loader = DataLoader(dev_data, batch_size=params.batch_size_dev, sampler=dev_sampler)

    best_acc = 0.0
    for epoch in range(params.epochs):
        tr_loss = train(model, optimizer, train_data_loader, epoch, loss_fn, device)

        correct, tr_loss_dev = test(model, dev_data_loader, device, loss_fn)
        acc = correct/len(dev_data)
        print(f"Epoch {epoch:02d}/{params.epochs} ======>   Acccuracy: {acc:02.4f}\
                Loss train: {tr_loss/len(train_data):04.4f}    Loss dev: {tr_loss_dev/len(dev_data):04.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./weights_best_acc_" + str(epoch+1) + ".pth")
        

        writer.add_scalar('training_loss', tr_loss/len(train_data), epoch+1)
        writer.add_scalar('dev_los', tr_loss_dev/len(dev_data), epoch+1)
        writer.add_scalar('dev_acc', acc, epoch+1)

    writer.close()


if __name__ == "__main__":
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--raw_data", default="./data_vi/train_data.json", type=str, help="Path of raw data")
    parsers.add_argument("--model_pretrained", default="bert-base-uncased", type=str, help="Path or string pretrained model")
    parsers.add_argument("--epochs", default=2, type=int, help="Number of epochs")
    parsers.add_argument("--batch_size_train", default=16, type=int, help="Batch size training set")
    parsers.add_argument("--batch_size_dev", default=16, type=int, help="Batch size dev set")


    
    params = parsers.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AcrBertModel.from_pretrained(params.model_pretrained).to(device=device)
    # model.save_pretrained("bert-base")
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

    writer = SummaryWriter()
    print("Loading.................")
    with open(params.raw_data, "r", encoding="UTF-8") as f:
        raw_data = json.load(f)

    main(model, raw_data, writer, params, device, optimizer, loss_fn)
    
