import argparse
import os
import json

import numpy as np

def normalize(list_token: list):
    """
    Function: Lower string
    """
    return list(map(lambda x: x.lower(), list_token))

def preprocessing(data: list, label: str):
    """
    Function: Preprocessing data
    args:
        data: Raw data
        label: pos or neg
    """
    for sample in data:
        sample['text'] = sample['text'].lower()
        if label == "pos": sample["label"] = 1
        else: sample["label"] = 0
    return data

def create_inputs_targets(examples):
    """
    Function: Create dictionary dataset for training
    """
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
        "label": []
    }
    for item in examples:
        if item.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
        
    X = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    Y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"], dataset_dict["label"]]
    return X, Y
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data_vi/vi_final_train_data_small.json", 
                        help= "Dataset for Acronym Disambiguation")
    parser.add_argument("--data_folder", type=str, default="./data_vi",
                        help= "Folder for sampling positive dataset")
    parser.add_argument("--mode", type=str, default="train",
                        help= "Mode of dataset")
    args = parser.parse_args()

    with open(args.data, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    if not os.path.isdir(args.data_folder): os.mkdir(args.data_folder)

    data_pos = preprocessing(data, "pos")
    print(data_pos[:2])

    with open(os.path.join(args.data_folder, f"{args.mode}_pos_data.json"), "w", encoding="UTF-8") as f:
        json.dump(data_pos, f)
