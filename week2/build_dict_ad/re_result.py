import json

with open("result_cxr/list_long_form_cxrv2.txt", "r") as f:
    dict_short_adcronym = {}
    dict_long_adcronym = {}
    i = 0 
    for line in f.readlines():
        if len(line.split(" ")) == 1: 
            dict_short_adcronym[f"None_{i}"] = [line]
            i += 1
        else:
            token = line.split(" ")
            acronym = "".join([j[0] for j in token])
            if acronym in dict_long_adcronym.keys():
                dict_long_adcronym[acronym].append(line)
            else: dict_long_adcronym[acronym] = [line]
with open("result_long.json", "w") as f:
    json.dump(dict_long_adcronym, f)
with open("result_short.json", "w") as f:
    json.dump(dict_short_adcronym, f)
