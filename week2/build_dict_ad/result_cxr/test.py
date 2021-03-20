import json
with open("list_long_form_cxrv2.txt", "r") as f:
    dict_file = {}
    for i, line in enumerate(f.readlines()):
        if len(line.split(" ")) == 1 : dict_file[f"None_{i}"] = line
        else:
            token = line.split(" ")
            acronym = "".join([j[0] for j in token])
            dict_file[acronym] = line
    # print(dict_file)
with open("dict_acronym_cxrv2.json", "w", encoding="utf8") as f:
        json.dump(dict_file, f)
        