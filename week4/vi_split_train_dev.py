import json
import argparse
from sklearn.model_selection import train_test_split
import random

def split_train_dev(data: list, test_size=0.1) -> list:
    """
    Function split data
    """
    train, dev = train_test_split(data, test_size=test_size)
    return train, dev

if __name__ == "__main__":
    with open("./data_vi/train_pos_data.json", "r", encoding="UTF-8") as f:
        data_pos = json.load(f)
    with open("./data_vi/train_neg_data.json", "r", encoding="UTF-8") as f:
        data_neg = json.load(f)
    
    data_pos.extend(data_neg)
    random.shuffle(data_pos)

    train, test = split_train_dev(data_pos)

    with open("./data_vi/train_data.json", "w", encoding="UTF-8") as f:
        json.dump(train, f)
    with open("./data_vi/test_data.json", "w", encoding="UTF-8") as f:
        json.dump(test, f)
    

