import numpy as np




def normalize(list_token):
    return list(map(lambda x: x.lower(), list_token))

def preprocessing(data, mode="train"):
    if mode in ["train", "val"]:
        for sample in data:
            sample["tokens"] = normalize(sample["tokens"])
            sample["text"] = " ".join(sample["tokens"])
            start_char_idx = 0
            for i in range(0, sample["acronym"]):
                start_char_idx += len(sample["tokens"][i])
                start_char_idx += 1
            sample["start_char_idx"] = start_char_idx
            sample["lenght_acronym"] = len(sample["tokens"][sample["acronym"]])
    else:
        return None
    return data

def create_inputs_targets(examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in examples:
        if item.skip is False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
        
    X = [dataset_dict["input_ids"], dataset_dict["token_type_ids"], dataset_dict["attention_mask"]]
    Y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return X, Y