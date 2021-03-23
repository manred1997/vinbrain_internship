import json
import re

def remove_string(string):
    if re.search("(xx+|\sx\s|x\.|^x\s|\sx$)", string):  return True
    elif re.search("^xx*\s", string): return True
    elif re.search("^\.\w", string):    return True
    else: return False

def filter_long_words(data):
    for i in range(20):
        for key in data:
            for value in data[key]:
                if remove_string(value):
                    # print(key)
                    # print(value)
                    data[key].remove(value)
    new_data = {}
    for key in data:
        if not data[key]: continue
        else: new_data[key] = data[key]
    return new_data

with open("./result_cxr/long_dict.json", "r") as f:
    data = json.load(f)

data = filter_long_words(data)

with open("./result_cxr/filtered_long_dict.json", "w") as f:
    json.dump(data, f)
