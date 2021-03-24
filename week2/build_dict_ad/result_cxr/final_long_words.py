import json
import re

data_cxr = []
with open("cxrv2.txt", "r") as f:
    for line in f.readlines():
        data_cxr.append(line)

with open("filtered_long_dict.json", "r") as f:
    long_dict_data = json.load(f)

no_filter_long_dict = {}
filter_long_dict = {}
for key, value in long_dict_data.items():
    if len(value) >= 5:
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

with open("final_long_dict.json", "w") as f:
    json.dump(filtered_long_dict, f)