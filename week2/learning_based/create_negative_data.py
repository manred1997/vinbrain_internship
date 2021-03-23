import random
import argparse

def negative_data(positive_data, diction):
    """
    Funciton: Create negative samples
    args:
        positive_data: training data whose format {
            'acronym': ...,
            'expansion': ...,
            'id': ..., 
            'tokens': ...,
            'text': ...,
            'start_char_idx: ...,
            'len_acronym': ...,
        }
        and
        diction: dictionary of acronym and able expansion respectively
    """

    neg_data = []
    tmp = 0
    for sample in positive_data:
        try:
            acronym = sample["tokens"][sample["acronym"]]
            list_neg_expansion = diction[acronym.upper()].copy()
            list_neg_expansion.remove(sample["expansion"])
            if len(list_neg_expansion) > 1: 
                list_neg_expansion = random.sample(list_neg_expansion, random.randint(1,2))
            for i in list_neg_expansion:
                neg_data.append(sample.copy())
                neg_data[tmp]["expansion"] = i
                tmp += 1
        except: continue
    
    return neg_data