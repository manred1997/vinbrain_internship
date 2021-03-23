import json
import re

def remove_string(string):
    if re.search("(xx+|\sx\s|x\.|^x\s|\sx$)", string):  return True
    elif re.search("^\.\w", string):    return True
    else: return False

def filter_long_words(data):
    for key in data.items():
        for value in data[key]:
            if remove_string(value):
                data[key].remove(value)
    new_data = {}
    for key in data:
        if not data[key]: continue
        else: new_data[key] = data[key]
    return new_data
