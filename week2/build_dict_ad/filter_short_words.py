import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open("./result_cxr/short_dict.json", "r") as f:
    data = json.load(f)
    length_words = {}
    for i in data:
        if str(len(data[i][0])) in length_words.keys():
            length_words[str(len(data[i][0]))] += 1
            # length_words[""]
        else:
            length_words[str(len(data[i][0]))] = 1
# print(data)


def help(i):
    if str(i) == "1":
        return length_words[str(i)]
    else:
        return length_words[str(i)] + help(i-1)
y = []
for i in range(1, 18):
    y.append(help(i))
x = list(range(1, 18))
assert len(x) == len(y)

fig = plt.figure(figsize=(8,10))
# plt.plot(x, y)
sns.barplot(x=x, y=y)
sns.lineplot(x=x, y=y)
plt.xlabel("Số ký tự")
plt.xticks(np.arange(1, 18))
plt.ylabel("Số từ")
plt.title("Số từ chứa nhiều hơn X ký tự")
# plt.savefig(f"./result_cxr/filter_short_acronym_cxr.png")
plt.show()

new_data = {}
for k, v in data.items():
    if len(v[0]) < 9: continue
    else: new_data[k] = v
with open("./result_cxr/filtered_short_dict.json", "w", encoding="utf8") as f:
    json.dump(new_data, f)
