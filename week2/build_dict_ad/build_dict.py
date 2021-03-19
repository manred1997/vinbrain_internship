import nltk
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, array.shape)

def POS(string):
    tokenized_string = nltk.word_tokenize(string)
    # print(tokenized_string)
    regex = r"""
  NP: {<JJ>*<NN|NNS|NNP>+}   # chunk determiner/possessive, adjectives and nouns
                # chunk sequences of proper nouns
"""
    chunkParser = nltk.RegexpParser(regex)

    ner = chunkParser.parse(nltk.pos_tag(tokenized_string))
    result = []
    for i in ner:
        try: 
            if i.label() == "NP": result.append(i)
            else: continue
        except: 
            continue
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./cxrv2.txt", help="Input file")
    parser.add_argument("--output_file", type=str, default="./list_long_form_cxrv2.txt", help="Output file")
    parser.add_argument("--dict_arc", type=str, default="./dict_acronym_cxrv2.json", help="Json file")
    parser.add_argument("--top_k", type=int, default=1000, help="Top k words that can be abbreviated")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        results = []
        for line in f.readlines():
            results.extend(POS(line))
    
    vocab = []
    for i in results:
        if len(i) == 1: vocab.append(i[0][0])
        else:
            tmp = []
            for j in i:
                tmp.append(j[0])
            vocab.extend([" ".join(tmp)])

    vocab = list(map(lambda x: x.lower(), vocab))
    unique = np.unique(vocab, return_counts=True)
    # top_k_indice = largest_indices(unique[1], 40)[0] # top 30
    # top_k_words = unique[0][[top_k_indice.tolist()]]

    sorted_unique = []
    for i in range(len(unique[0])):
        sorted_unique.append({
            "freq": unique[1][i],
            "unique": unique[0][i]
        })
    sorted_unique = sorted(sorted_unique, key= lambda x: x["freq"], reverse=True)
    x = list(range(1, len(sorted_unique)+1))
    y = []
    tmp = 0
    for i in sorted_unique:
        tmp += i["freq"]
        y.append(tmp)

    fig = plt.figure(figsize=(8,10))
    # plt.plot(x, y)
    sns.barplot(x=x, y=y)
    sns.lineplot(x=x, y=y)
    plt.xlabel("Top - k")
    plt.xticks(np.linspace(1,len(sorted_unique), 10, dtype=int))
    plt.ylabel("Number of words")
    plt.title("Histogram of words that can be abbreviated")
    plt.savefig("histogram_eng_arconym.png")
    plt.show()

    # top_k_words = []
    # for i in range(args.top_k):
    #     top_k_words.append(sorted_unique[i]["unique"])

    # with open(args.output_file, "w") as f:
    #     f.write("\n".join(top_k_words))
    # dict_file = {}
    # for i, value in enumerate(top_k_words):
    #     if len(value.split(" ")) == 1 : dict_file[f"None_{i}"] = value
    #     else:
    #         token = value.split(" ")
    #         acronym = "".join([j[0] for j in token])
    #         dict_file[acronym] = value
    # # print(dict_file)
    # with open(args.dict_arc, "w") as f:
    #     json.dump(dict_file, f)
    with open("tmp.txt") as f:
        for item in sorted_unique:
            f.write("%s\n" % item)