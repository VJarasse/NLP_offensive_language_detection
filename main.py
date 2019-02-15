import pandas as pd
import numpy as np
from corpus import *
from neuralnet import *
from utils import *
import os

# Loading the dataset
dataframe = pd.read_csv('data/offenseval-training-v1.tsv', sep="\t", header=0)


# Use once to save the clean vocabulary to a file

corpus = Corpus(dataframe, 'tweet')
with open("vocabulary.txt", "w") as f:
    for voc in corpus.clean_vocabulary:
        f.write(voc + "\n")
with open("corpus_tweets.txt", "w") as f:
    for sentence in corpus.tokenized_corpus:
        for token in sentence:
            f.write(token + " ")
        f.write("\n")

# Get the vocabulary back
vocabulary = []
with open("vocabulary.txt", "r") as file:
        vocabulary = file.read().splitlines()

print("Length of vocabulary is : ", len(vocabulary), " words and expressions")

# Global variables for the NN and the training
# we will train for N epochs (N times the model will see all the data)
epochs = 20
# the input dimension is the vocabulary size
INPUT_DIM = len(vocabulary)
# we define our embedding dimension (dimensionality of the output of the first layer)
EMBEDDING_DIM = 100
# dimensionality of the output of the second hidden layer
HIDDEN_DIM = 50
# the output dimension is the number of classes, 1 for binary classification
OUTPUT_DIM = 1
#
sent_lengths = [len(sent) for sent in tokenized_corpus]

word2idx = get_word2idx(vocabulary)






def get_model_inputs(tokenized_corpus, word2idx, labels, max_len):
        # we index our sentences
        vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]

        # we create a tensor of a fixed size filled with zeroes for padding

        sent_tensor = Variable(torch.zeros((len(vectorized_sents), max_len))).long()

        sent_lengths = [len(sent) for sent in vectorized_sents]

        # we fill it with our vectorized sentences

        for idx, (sent, sentlen) in enumerate(zip(vectorized_sents, sent_lengths)):
                sent_tensor[idx, :sentlen] = torch.LongTensor(sent)

        label_tensor = torch.FloatTensor(labels)

        return sent_tensor, label_tensor



max_len = np.max(np.array(sent_lengths))


train_sent_tensor, train_label_tensor = get_model_inputs(tokenized_corpus, word2idx, train_labels, max_len)

print(train_sent_tensor.shape)
