import random
import argparse
import os
import json

from vi_preprocessing import preprocessing

def negative_data(positive_data:list, diction:dict) -> list:
    """
    Funciton: Create negative samples
    args:
        positive_data: training data whose format {
            'acronym': ...,
            'expansion': ...,
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
            acronym = sample["text"][sample["start_char_idx"]:sample["start_char_idx"]+sample['length_acronym']]
            list_neg_expansion = diction[acronym].copy()
            list_neg_expansion.remove(sample["expansion"])
            if len(list_neg_expansion) > 1: 
                list_neg_expansion = random.sample(list_neg_expansion, random.randint(1,2))
            for i in list_neg_expansion:
                neg_data.append(sample.copy())
                neg_data[tmp]["expansion"] = i
                neg_data[tmp]["label"] = 0
                tmp += 1
        except: continue
    
    return neg_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pos", type=str, default="./data_vi/train_pos_data.json", 
                        help= "Positive dataset for Acronym Disambiguation")
    parser.add_argument("--diction", type=str, default="./data_vi/vi_final_long_dict_fix.json",
                        help= "Dictionary dataset for Acronym Disambiguation")
    parser.add_argument("--data_neg", type=str, default="./data_vi",
                        help= "Folder for sampling negative dataset")
    parser.add_argument("--mode", type=str, default="train",
                        help= "Mode of dataset")

    args = parser.parse_args()

    with open(args.data_pos, "r", encoding="utf8") as f:
        data_pos = json.load(f)
    with open(args.diction, "r", encoding="utf8") as f:
        diction = json.load(f)
    
    if not os.path.isdir(args.data_neg): os.mkdir(args.data_neg)

    # data_pos = preprocessing(data_pos, args.mode, "neg")
    data_neg = negative_data(data_pos, diction)
    print(f"Number of samples: {len(data_neg)}")

    with open(os.path.join(args.data_neg, f"{args.mode}_neg_data.json"), "w", encoding="UTF-8") as f:
        json.dump(data_neg, f)

