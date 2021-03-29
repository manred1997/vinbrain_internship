import json
import argparse

import torch
import numpy as np

from model import AcrBertModel
from tokenizers import BertWordPieceTokenizer


class AcronymExpansionModel:
    def __init__(self, acronym_long_dict, acronym_short_dict=None, model=None, config, vocab):
        self.acn_dict = self.load_dict(acronym_long_dict, acronym_short_dict)
        self.model = AcrBertModel(model, config)
        self.tokenizer = BertWordPieceTokenizer(vocab, lowercase=True)

    def expand_acronym(self, text):
        def get_acr(text):
            if text[-1] == ' ':
                return None
            return text.split(' ')[-1]
        
        acr = get_acr(text)
        if acr is None:
            return []

        full_texts = self.select(acr, text)
        return full_texts
    
    def load_dict(self, acronym_long_dict, acronym_short_dict):
        """
        MODIFY this
        
        load and preprocess the acronym dict
        """
        with open(acronym_long_dict, 'r', encoding='utf8') as f:
            acronym_dict = json.load(f)
        if acronym_short_dict is not None:
            with open(acronym_short_dict, 'r', encoding='utf8') as f:
                data = json.load(f)
                for key, value in data.items():
                    try: acronym_dict[key].extend(value)
                    except: acronym_dict[key] = value
        return acronym_dict

    def preprocessing(self, acronym, text)
        start_char_idx = text.find(acronym)
        end_char_idx = start_char_idx + len(acronym)
        tokenized_context = self.tokenizer.encode(text)
        is_char_in_context = [0]*len(text)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_context[idx] = 1
        arc_token_idx  = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_context[start:end]) > 0: arc_token_idx.append(idx)
        start_token_idx = torch.tensor(arc_token_idx[0], dtype=torch.int64)
        end_token_idx = torch.tensor(arc_token_idx[-1], dtype=torch.int64)

        return tokenized_context, start_token_idx, end_token_idx

    def predict(self, input_ids, token_type_ids, attention_mask, start_token_idx, end_token_idx):
        prop, _ = self.model(input_ids,
                                token_type_ids,
                                attention_mask,
                                start_token_idx,
                                end_token_idx)
        return prop
    

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
            if self.acn_dict.get(acronym, ""):
                tokenized_context, start_token_idx, end_token_idx = self.preprocessing(acronym, text)
                list_expansion = self.acn_dict.get(acronym, "")
                props = []
                for expansion in list_expansion:
                    tokenized_expansion = self.tokenizer.encode(expansion)
                    input_ids = tokenized_context.ids + tokenized_expansion.ids[1:]
                    token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_expansion.ids[1:])
                    attention_mask = [1] * len(input_ids)
                    padding_length = 384 - len(input_ids)
                    if padding_length > 0:
                        input_ids = input_ids + ([0]* padding_length)
                        token_type_ids = token_type_ids + ([0]* padding_length)
                        attention_mask = attention_mask + ([0]* padding_length)

                    input_ids = torch.tensor(input_ids, dtype=torch.int64)
                    token_type_ids = torch.tensor(token_type_ids, dtype=torch.int64)
                    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
                    props.append(int(self.predict(input_ids, token_type_ids, attention_mask, start_token_idx, end_token_idx)))
                return list_expansion[int(np.argmax(props))]

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