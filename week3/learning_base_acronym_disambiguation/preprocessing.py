import argparse
import os
import json

import numpy as np

def normalize(list_token: list):
    """
    Function: Lower string
    """
    return list(map(lambda x: x.lower(), list_token))

def preprocessing(data: list, mode: list, label: str):
    """
    Function: Preprocessing data
    args:
        data: Raw data
        mode: train or dev
        label: pos or neg
    """
    if mode in ["train", "dev"]:
        for sample in data:
            sample["tokens"] = normalize(sample["tokens"])
            sample["text"] = " ".join(sample["tokens"])
            start_char_idx = 0
            for i in range(0, sample["acronym"]):
                start_char_idx += len(sample["tokens"][i])
                start_char_idx += 1
            sample["start_char_idx"] = start_char_idx
            sample["lenght_acronym"] = len(sample["tokens"][sample["acronym"]])
            if label == "pos": sample["label"] = 1
            else: sample["label"] = 0
    else:
        return None
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
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
        
    X = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    Y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"], dataset_dict["label"]]
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../AAAI-21-SDU-shared-task-2-AD/dataset/train.json", 
                        help= "Dataset for Acronym Disambiguation")
    parser.add_argument("--data_folder", type=str, default="./pos_data",
                        help= "Folder for sampling positive dataset")
    parser.add_argument("--mode", type=str, default="train",
                        help= "Mode of dataset")
    args = parser.parse_args()

    with open(args.data, "r", encoding="UTF-8") as f:
        data = json.load(f)
    
    if not os.path.isdir(args.data_folder): os.mkdir(args.data_folder)

    data_pos = preprocessing(data, args.mode, "pos")

    with open(os.path.join(args.data_folder, f"{args.mode}_pos_data.json"), "w", encoding="UTF-8") as f:
        json.dump(data_pos, f)
