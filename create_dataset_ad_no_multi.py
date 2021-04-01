import json
import re

from multiprocessing import cpu_count, Pool

procs = cpu_count()

  
def chunk(list_text, chunk_len):
    for i in range(0, len(list_text), chunk_len):
        yield list_text[i: i + chunk_len]

def get_data(acronym, expansion, line):
    if re.search(expansion, line):
        if len(re.finditer(expansion, line)) > 1:  return {}
        else:
            start_char_idx = list(re.finditer(expansion, line))[0].span()[0]
            len_acronym = len(acronym)
            line = re.sub(expansion, acronym, line)
            return {
                "text": line,
                "start_char_idx": start_char_idx,
                "length_acronym": len_acronym,
                "expansion": expansion
            }
    else: return {}

def build_dataset(data_info):
    tmp = []
    for line in data_info['lines']:
        try:
            tmp.append(get_data(data_info['acronym'], data_info['expansion'], line))
        except: continue
    return tmp
    



with open("./result_vn_2/final_long_dict.json", "r", encoding="UTF-8") as f:
    diction = json.load(f)

with open("./data/cleaned_data.txt", "r", encoding="UTF-8") as f:
    data = []
    for line in f.readlines():
        data.append(line)



dataset = []
num_line_per_proc = len(data) // procs

for acronym, expansions in diction.items():
    for expansion in expansions:
        chunked_lists = list(chunk(data, num_line_per_proc))
        data_info = []
        for acr_exp, chunked_list in zip((acronym, expansion), chunked_lists):

            chunk_info = {
                'lines': chunked_list,
                'acronym': acr_exp[0],
                'expansion': acr_exp[1]
            }

            data_info.append(chunk_info)
            
        pool = Pool(processes=procs)
        dataset.extend(pool.map(build_dataset, data_info))

with open("train.json", "w", encoding="UTF-8") as f:
    json.dump(dataset, f)


