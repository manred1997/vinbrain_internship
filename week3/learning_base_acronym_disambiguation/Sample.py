from tqdm import tqdm
import sys
from colorama import Fore

class Sample:
    def __init__(self, tokenizer, expansion, context, start_char_idx, len_acronym, label, max_seq_lenght=384):
        self.tokenizer = tokenizer #tokenizer BertWordPieceTokenizer
        self.expansion = expansion
        self.context = context
        self.start_char_idx = start_char_idx
        self.len_acronym = len_acronym
        self.max_seq_lenght = max_seq_lenght
        self.skip = False
        
        self.start_token_idx = -1
        self.end_token_idx = -1

        self.label = int(label)
        
    def preprocess(self):
        tokenized_expansion = self.tokenizer.encode(self.expansion)
        tokenized_context = self.tokenizer.encode(self.context)
        
        end_char_idx = self.start_char_idx + self.len_acronym
        if end_char_idx >= len(self.context): 
            self.skip = True
            return
        
        is_char_in_context = [0]*len(self.context)
        for idx in range(self.start_char_idx, end_char_idx):
            is_char_in_context[idx] = 1
        
        arc_token_idx  = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_context[start:end]) > 0: arc_token_idx.append(idx)
        if len(arc_token_idx) == 0:
            self.skip = True
            return
        self.start_token_idx = arc_token_idx[0]
        self.end_token_idx = arc_token_idx[-1]
        
        input_ids = tokenized_context.ids + tokenized_expansion.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_expansion.ids[1:])
        attention_mask = [1] * len(input_ids)
        
        
        padding_length = self.max_seq_lenght - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0]* padding_length)
            token_type_ids = token_type_ids + ([0]* padding_length)
            attention_mask = attention_mask + ([0]* padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask


def create_examples(raw_data, desc, tokenizer):
    p_bar = tqdm(total=len(raw_data), desc=desc,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    examples = []
    for item in raw_data:
        expansion = item["expansion"]
        context = item["text"]
        start_char_idx = item["start_char_idx"]
        lenght_acronym = item["lenght_acronym"]
        label = item["label"]
        example = Sample(tokenizer, expansion, context, start_char_idx, lenght_acronym, label)
        example.preprocess()
        examples.append(example)
        p_bar.update(1)
    p_bar.close()
    return examples