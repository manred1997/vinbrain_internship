import nltk
import numpy as np
import argparse
# from util.crf.pos_tag import pos_tag # refe to https://github.com/undertheseanlp/pos_tag
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

def POS_vn(text):

    regex = """
    NP: {<N|Ni|Np|NNP>+}
    """
    chunkParser = nltk.RegexpParser(regex)
    token_pos = pos_tag(text)
    ner = chunkParser.parse(token_pos)
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
    parser.add_argument("--input_file", type=str, default="./result_cxr/cxrv2.txt", help="Input file")
    parser.add_argument("--top_k", type=int, default=10000, help="Top k words that can be abbreviated")
    parser.add_argument("--mode", type=str, default="cxr")
    args = parser.parse_args()

################## Part of Speech Tagging ##################
    with open(args.input_file, 'r', encoding="utf8") as f:
        results = []
        for line in f.readlines():
            if args.mode == "vn":
                results.extend(POS_vn(line))
            else:
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
    del vocab
    # top_k_indice = largest_indices(unique[1], 40)[0] # top 30
    # top_k_words = unique[0][[top_k_indice.tolist()]]

################## Sort Unique ##################
    sorted_unique = []
    sorted_short_unique = []
    sorted_long_unique = []
    for i in range(len(unique[0])):
        sorted_unique.append({
            "freq": unique[1][i],
            "unique": unique[0][i]
        })
        if len(unique[0][i].split(" ")) == 1:
            sorted_short_unique.append({
                "freq": unique[1][i],
                "unique": unique[0][i]
            })
        else:
            sorted_long_unique.append({
                "freq": unique[1][i],
                "unique": unique[0][i]
            })
    sorted_unique = sorted(sorted_unique, key= lambda x: x["freq"], reverse=True)
    sorted_short_unique = sorted(sorted_short_unique, key= lambda x: x["freq"], reverse=True)
    sorted_long_unique = sorted(sorted_long_unique, key= lambda x: x["freq"], reverse=True)
    del unique

################## PLOT HISTOGRAM ##################
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
    plt.savefig(f"./result_{args.mode}/histogram_all_acronym_{args.mode}.png")
    # plt.show()
    del x, y, fig

    x = list(range(1, len(sorted_long_unique)+1))
    y = []
    tmp = 0
    for i in sorted_long_unique:
        tmp += i["freq"]
        y.append(tmp)

    fig = plt.figure(figsize=(8,10))
    # plt.plot(x, y)
    sns.barplot(x=x, y=y)
    sns.lineplot(x=x, y=y)
    plt.xlabel("Top - k")
    plt.xticks(np.linspace(1,len(sorted_long_unique), 10, dtype=int))
    plt.ylabel("Number of words")
    plt.title("Histogram of words that can be abbreviated")
    plt.savefig(f"./result_{args.mode}/histogram_long_acronym_{args.mode}.png")
    # plt.show()
    del x, y, fig

    x = list(range(1, len(sorted_short_unique)+1))
    y = []
    tmp = 0
    for i in sorted_short_unique:
        tmp += i["freq"]
        y.append(tmp)

    fig = plt.figure(figsize=(8,10))
    # plt.plot(x, y)
    sns.barplot(x=x, y=y)
    sns.lineplot(x=x, y=y)
    plt.xlabel("Top - k")
    plt.xticks(np.linspace(1,len(sorted_short_unique), 10, dtype=int))
    plt.ylabel("Number of words")
    plt.title("Histogram of words that can be abbreviated")
    plt.savefig(f"./result_{args.mode}/histogram_short_acronym_{args.mode}.png")
    # plt.show()
    del x, y, fig

    


# ################## Write File ##################
    top_k_all_words = []
    for i in range(args.top_k):
        top_k_all_words.append(sorted_unique[i]["unique"])
    top_k_long_words = []
    for i in range(5000):
        top_k_long_words.append(sorted_long_unique[i]["unique"])
    top_k_short_words = []
    for i in range(1000):
        top_k_short_words.append(sorted_short_unique[i]["unique"])
    
    del sorted_unique, sorted_short_unique, sorted_long_unique


    with open(f"./result_{args.mode}/all_expansion.txt", "w", encoding="utf8") as f:
        f.write("\n".join(top_k_all_words))
    with open(f"./result_{args.mode}/long_expansion.txt", "w", encoding="utf8") as f:
        f.write("\n".join(top_k_long_words))
    with open(f"./result_{args.mode}/short_expansion.txt", "w", encoding="utf8") as f:
        f.write("\n".join(top_k_short_words))

    dict_all_adcronym = {}
    dict_long_adcronym = {}
    dict_short_adcronym = {}

    tmp = 0
    for word in top_k_all_words:
        if len(word.split(" ")) == 1: 
            dict_all_adcronym[f"None_{tmp}"] = [word]
            tmp += 1
        else:
            token = word.split(" ")
            acronym = "".join([j[0] for j in token])
            if acronym in dict_all_adcronym.keys():
                dict_all_adcronym[acronym].append(word)
            else: dict_all_adcronym[acronym] = [word]
    del top_k_all_words

    for word in top_k_long_words:
        token = word.split(" ")
        acronym = "".join([j[0] for j in token])
        if acronym in dict_long_adcronym.keys():
            dict_long_adcronym[acronym].append(word)
        else: dict_long_adcronym[acronym] = [word]
    del top_k_long_words

    tmp = 0
    for word in top_k_short_words:
        dict_short_adcronym[f"None_{tmp}"] = [word]
        tmp += 1
    del top_k_short_words
    
    with open(f"./result_{args.mode}/all_dict.json", "w", encoding="utf8") as f:
        json.dump(dict_all_adcronym, f)
    with open(f"./result_{args.mode}/long_dict.json", "w", encoding="utf8") as f:
        json.dump(dict_long_adcronym, f)
    with open(f"./result_{args.mode}/short_dict.json", "w", encoding="utf8") as f:
        json.dump(dict_short_adcronym, f)