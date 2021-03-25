import json
import re

data_cxr = []
with open("../data/cleaned_data.txt", "r", encoding="utf8") as f:
    for line in f.readlines():
        data_cxr.append(line)

with open("./result_vn_2/long_dict.json", "r", encoding="utf8") as f:
    long_dict_data = json.load(f)

for key, value in long_dict_data.items():
    value = sorted(value)
    filtered = []
    for i, c in enumerate(value):
        if (i+1) < len(value) and c in value[i+1]:  continue
        filtered.append(c)
    long_dict_data[key] = filtered

no_filter_long_dict = {}
filter_long_dict = {}
for key, value in long_dict_data.items():
    if len(value) >= 4:
        filter_long_dict[key] = value
    else: no_filter_long_dict[key] = value
freq_expansion = []
for key, value in filter_long_dict.items():
    print(f"START ============={key}=============")
    for pattern in value:
        # print(i)
        print(f"START ============={pattern}=============")
        tmp = 0
        for line in data_cxr:
            tmp += len([match.span() for match in re.finditer(pattern, line.lower())])
        print(f"Temp is: {tmp}")
        freq_expansion.append({
            key: { "expansion": pattern,
                    "freq": tmp
            }
        })
        print(f"END ============={pattern}=============")
    print(f"END ============={key}=============")

filtered_long_dict = {}
for key in list(filter_long_dict.keys()):
    # print(i)
    tmp = []
    for i in freq_expansion:
        if key in list(i.keys()):
            tmp.append(i[key])
    # print(tmp)
    tmp = sorted(tmp, key=lambda x: x["freq"], reverse=True)
    # tmp = tmp[:5]
    # print(tmp)
    filtered_long_dict[key] =[i["expansion"] for i in tmp]

filtered_long_dict.update(no_filter_long_dict)

tmp = {}
for key, value in filtered_long_dict.items():
    tmp_2 = value
    for sample in tmp_2:
        token = sample.split(" ")
        len_token = [len(s) for s in token]
        if sum([1 for i in len_token if i < 2]) > 0:
            tmp_2.remove(sample)
    if tmp_2:
        tmp[key] = tmp_2
filtered_long_dict = tmp

with open("./result_vn_2/final_long_dict.json", "w", encoding="utf8") as f:
    json.dump(filtered_long_dict, f)