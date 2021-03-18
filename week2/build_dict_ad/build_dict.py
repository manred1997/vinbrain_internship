import nltk
import numpy as np

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

if __name__ == "__main__":
    with open('english_clean.txt', 'r') as f:
        results = []
        for line in f.readlines():
            results.extend(NER(line))
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
    unique = np.unique(vocab, return_counts=True)
    top_k_indice = largest_indices(unique[1], 30)[0] # top 30
    top_k_words = unique[0][[top_k_indice.tolist()]]
    print(top_k_words)