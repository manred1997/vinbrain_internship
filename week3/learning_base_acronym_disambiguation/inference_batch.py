import json
import argparse

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import numpy as np

from model import AcrBertModel
from tokenizers import BertWordPieceTokenizer

import time

class AcronymExpansionModel:
    def __init__(self, acronym_long_dict="../AAAI-21-SDU-shared-task-2-AD/dataset/diction.json", 
                        model="./bert_base/weights_epoch51.pth",
                        config= "./bert_base/config.json",
                        vocab="./slow_token/vocab.txt"):
        self.acn_dict = self.load_dict(acronym_long_dict)
        self.model = AcrBertModel.from_pretrained(pretrained_model_name_or_path=model, 
                                                    config=config)
        # print(self.model)
        self.tokenizer = BertWordPieceTokenizer(vocab, lowercase=True)

    def expand_acronym(self, text):
        def get_acr(text):
            if text[-1] == ' ':
                return None
            return text.split(' ')[-1]
        
        acr = get_acr(text)
        if acr is None:
            return []
        # print(acr)

        full_texts = self.select(acr, text)
        return full_texts
    
    def load_dict(self, acronym_long_dict):
        """
        MODIFY this
        
        load and preprocess the acronym dict
        """
        with open(acronym_long_dict, 'r', encoding='utf8') as f:
            acronym_dict = json.load(f)
        # if acronym_short_dict is not None:
        #     with open(acronym_short_dict, 'r', encoding='utf8') as f:
        #         data = json.load(f)
        #         for key, value in data.items():
        #             try: acronym_dict[key].extend(value)
        #             except: acronym_dict[key] = value
        return acronym_dict
    
    def preprocessing(self, acronym, text, expansions):
        start_char_idx = text.find(acronym)
        end_char_idx = start_char_idx + len(acronym)
        tokenized_context = self.tokenizer.encode(text)
        is_char_in_context = [0]*len(text)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_context[idx] = 1
        arc_token_idx  = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_context[start:end]) > 0: arc_token_idx.append(idx)
        start_token_idx = torch.tensor([arc_token_idx[0]]*len(expansions), dtype=torch.int64)
        end_token_idx = torch.tensor([arc_token_idx[-1]]*len(expansions), dtype=torch.int64)
        input_ids = []
        token_type_ids = []
        attention_masks = []
        for expansion in expansions:
            tokenized_expansion = self.tokenizer.encode(expansion)
            input_id = tokenized_context.ids + tokenized_expansion.ids[1:]
            token_type_id = [0]*len(tokenized_context.ids) + [1]*len(tokenized_expansion.ids[1:])
            attention_mask = [1]*len(input_id)
            padding_length = 384 - len(input_id)
            if padding_length > 0:
                input_id = input_id + ([0]* padding_length)
                token_type_id = token_type_id + ([0]* padding_length)
                attention_mask = attention_mask + ([0]* padding_length)
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)
        return TensorDataset(torch.tensor(input_ids, dtype=torch.int64),
                            torch.tensor(token_type_ids, dtype=torch.int64),
                            torch.tensor(attention_masks, dtype=torch.float),
                            start_token_idx,
                            end_token_idx)
                      

    def select(self, acronym, text):
        """
        MODIFY this
        
        select the full phrase from 
        an acronym in a list of options
        """
        if self.model is None:
            if self.acn_dict.get(acronym, ""):
                return self.acn_dict.get(acronym)[:min(5, len(self.acn_dict.get(acronym)))]
            else: return ""
        else:
            if self.acn_dict.get(acronym.upper(), ""):
                expansions = self.acn_dict.get(acronym.upper(), "")
                inference_data = self.preprocessing(acronym, text, expansions)
                inference_sampler = SequentialSampler(inference_data)
                inference_loader = DataLoader(inference_data, batch_size=32, sampler=inference_sampler)
                self.model.eval()
                start = time.time()
                with torch.no_grad():
                    for batch in inference_loader:
                        input_word_ids, input_type_ids, input_mask, start_token_idx, end_token_idx = batch
                        props, _ = self.model(input_ids=input_word_ids,
                                                token_type_ids=input_type_ids,
                                                attention_mask=input_mask,
                                                start_token_idx=start_token_idx,
                                                end_token_idx=end_token_idx)
                print(f"inference time: {time.time() - start}")
                return  expansions[int(torch.argmax(props))]

            else: return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input text for model acronym disambiguation")
    parser.add_argument("--acr_long_dict", type=str, default="./result_vn_2/final_long_dict.json", help="Acronym dictionary")
    parser.add_argument("--acr_short_dict", type=str, default="./result_vn_2/short_dict.json", help="Acronym dictionary")
    parser.add_argument("--model", type=str, default=None, help="Model for acronym disambiguation")

    args = parser.parse_args()

    AcronymExpansion = AcronymExpansionModel(args.acr_long_dict, args.acr_short_dict, args.model)
    
    expansion_acr = AcronymExpansion.expand_acronym(args.text)
    print(expansion_acr)