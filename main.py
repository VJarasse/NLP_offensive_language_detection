import pandas as pd
import numpy as np
from corpus import *
from neuralnet import *
from utils import *

# Loading the dataset
dataframe = pd.read_csv('data/offenseval-training-v1.tsv', sep="\t", header=0)

# Use once to save the clean vocabulary to a file
'''
corpus = Corpus(dataframe, 'tweet')
with open("vocabulary.txt", "w") as f:
    for voc in corpus.clean_vocabulary:
        f.write(voc + "\n")
with open("corpus_tweets.txt", "w") as f:
    for sentence in corpus.tokenized_corpus:
        for token in sentence:
            f.write(token + " ")
        f.write("\n")
'''
# Get the vocabulary back
with open("data/vocabulary.txt", "r") as file:
    vocabulary = file.read().splitlines()
# Get the cleaned tokenized corpus back
with open("data/corpus_tweets.txt", "r") as file:
    tmp = file.read().splitlines()
# In each sentence, we get rid of the last token, which is '\n'
tokenized_corpus = [[token for token in sentence.split(' ')][:-1] for sentence in tmp]

print("Length of vocabulary is : ", len(vocabulary), " words and expressions")

# Global variables for the NN and the training
# we will train for N epochs (N times the model will see all the data)
epochs = 20
# the input dimension is the vocabulary size
INPUT_DIM = len(vocabulary)
# we define our embedding dimension (dimensionality of the output of the first layer)
EMBEDDING_DIM = 10000
# dimensionality of the output of the second hidden layer
HIDDEN_DIM = 500
# the output dimension is the number of classes, 1 for binary classification
OUTPUT_DIM = 1
#
sent_lengths = [len(sent) for sent in tokenized_corpus]
max_len = np.max(np.array(sent_lengths))

word2idx = get_word2idx(vocabulary)

train_sent_tensor, train_label_tensor = get_model_inputs(tokenized_corpus, word2idx, train_labels, max_len)

print(train_sent_tensor.shape)
