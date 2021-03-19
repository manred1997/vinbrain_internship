import nltk
import numpy as np
import argparse
from util.crf.pos_tag import pos_tag # refe to https://github.com/undertheseanlp/pos_tag
import json
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

def NER(string):
    tokenized_string = nltk.word_tokenize(string)
    # print(tokenized_string)
    regex = r"""
  NP: {<DT>?<JJ>*<NN|NNS>}   # chunk determiner/possessive, adjectives and nouns
      {<NNP>+}                # chunk sequences of proper nouns
"""
    chunkParser = nltk.RegexpParser(regex)
    # print(tokenized_string)
    # print(nltk.pos_tag(tokenized_string))
    ner = chunkParser.parse(nltk.pos_tag(tokenized_string))
    # print(ner)
    result = []
    for i in ner:
        # print(i.label())
        try: 
            if i.label() == "NP": result.append(i)
            # elif i.label() == "NNP": result.append(i)
            # elif i.label() == "NN": result.append(i)
            else: continue
        except: 
            continue
    return result

def NER_vn(text):

    regex = """
    NP: {<Ab>?<A>*<Nb|N>+}
    {<N>+}
    """
    chunkParser = nltk.RegexpParser(regex)
    token_pos = pos_tag(text)
    ner = chunkParser.parse(token_pos)
    result = []
    for i in ner:
        # print(i.label())
        try: 
            if i.label() == "NP": result.append(i)
            # elif i.label() == "NNP": result.append(i)
            # elif i.label() == "NN": result.append(i)
            else: continue
        except: 
            continue
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../vietnam_clean.txt", help="Input file")
    parser.add_argument("--output_file", type=str, default="../list_long_form_vn.txt", help="Output file")
    parser.add_argument("--dict_arc", type=str, default="../dict_acronym_vn.json", help="Json file")
    args = parser.parse_args()


    with open(args.input_file, 'r', encoding="utf8") as f:
        results = []
        for line in f.readlines():
            results.extend(NER_vn(line))
        # print(results)
    # print(len(results[13]))
    vocab = []
    for i in results:
        if len(i) == 1: vocab.append(i[0][0])
        else:
            tmp = []
            for j in i:
                tmp.append(j[0])
            # print(tmp)
            vocab.extend([" ".join(tmp)])
    # print(vocab)
    vocab = list(map(lambda x: x.lower(), vocab))
    # print(vocab)
    unique = np.unique(vocab, return_counts=True)
    top_k_indice = largest_indices(unique[1], 40)[0] # top 30
    top_k_words = unique[0][[top_k_indice.tolist()]]
    # print(top_k_words)
    with open(args.output_file, "w", encoding="utf8") as f:
        f.write("\n".join(top_k_words))
    dict_file = {}
    for i in top_k_words:
        if len(i.split(" ")) == 1 : dict_file[i] = None
        else:
            token = i.split(" ")
            acronym = "".join([j[0] for j in token])
            dict_file[i] = acronym
    # print(dict_file)
    with open(args.dict_arc, "w") as f:
        json.dump(dict_file, f)