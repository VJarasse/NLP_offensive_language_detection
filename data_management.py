import pandas as pd
import numpy as np
from corpus import *

# Loading the dataset

def corpus_and_voc():
    dataframe = pd.read_csv('data/offenseval-training-v1.tsv', sep="\t", header=0)
    label1 = dataframe['subtask_a']
    label2 = dataframe['subtask_b']
    label3 = dataframe['subtask_c']

    # Use once to save the clean vocabulary
    corpus = Corpus(dataframe, 'tweet')
    with open("data/vocabulary.txt", "w") as f:
        for voc in corpus.clean_vocabulary:
            f.write(voc + "\n")

    with open("data/corpus_tweets.txt", "w") as f:
        for sentence in corpus.tokenized_corpus:
            for token in sentence:
                f.write(token + " ")
            f.write("\n")

    with open("data/labels_1.txt", "w") as f:
        for label in label1:
            if label == "OFF":
                f.write("1\n")
            else:
                f.write("0\n")

    with open("data/labels_2.txt", "w") as f:
        for label in label2:
            if label == "TIN":
                f.write("1\n")
            else:
                f.write("0\n")

    with open("data/labels_3.txt", "w") as f:
        for label in label3:
            if label == "IND":
                f.write("1\n")
            elif label == "GRP":
                f.write("2\n")
            elif label == "OTH":
                f.write("3\n")
            else:
                f.write("0\n")


def embedding():
    # Get back clean vocabulary from txt file
    with open("data/vocabulary.txt", "r") as file:
        vocabulary = file.read().splitlines()

    # Keep track of unknown words for later processing
    found = 0.
    voc_size = len(vocabulary)

    with open("data/emb_dic.txt", "w") as f, open("data/glove.twitter.27B.100d.txt", "r") as glove:
        # Create dictionary of all glove embedding from txt file
        emb_dict = {}
        rejected = []
        for line in glove:
            values = line.split()
            word = values[0]
            if word in vocabulary:
                found += 1
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    f.write(line)
                except:
                    print("Parsing problem on word ", word, " discarding it")
        glove.close()
    ratio = found / voc_size * 100
    return ratio


#corpus_and_voc()

ratio = embedding()
print("Ratio of found words ", ratio)

