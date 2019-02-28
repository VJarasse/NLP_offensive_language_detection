import pandas as pd
import numpy as np

from corpus import *
from neuralnet import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

#  <-------------- Dataset management -------------> #

# Get the vocabulary back
with open("data/vocabulary.txt", "r") as file:
    vocabulary = file.read().splitlines()

# Get the cleaned tokenized corpus back
with open("data/corpus_tweets.txt", "r") as file:
    tmp = file.read().splitlines()

# In each sentence, we get rid of the last token, which is '\n'
clean_corpus = [sentence[:-1] for sentence in tmp]

tokenized_corpus = [[token for token in sentence.split(' ')][:-1] for sentence in tmp]

# Get the labels
with open("data/labels_1.txt", "r") as file:
    label1 = file.read().splitlines()
with open("data/labels_2.txt", "r") as file:
    label2 = file.read().splitlines()
with open("data/labels_3.txt", "r") as file:
    label3 = file.read().splitlines()

label1 = [float(i) for i in label1]
label2 = [float(i) for i in label2]
label3 = [float(i) for i in label3]

#  <-------------- END Dataset management -------------> #

#  <-------------- Word embedding management -------------> #

embedding_path = 'data/emb_dict.txt'
emb_dict = {}
glove = open(embedding_path)
for line in glove:
    values = line.split()
    word = values[0]
    try:
        vector = np.asarray(values[1:], dtype='float32')
        print(vector)
        emb_dict[word] = vector
    except:
        print("Parsing problem on word ", word, " discarding it")
glove.close()

#  <-------------- END Word embedding management -------------> #

#  <-------------- Tweets corpus preparation for NN -------------> #

def embed_corpus(emb_dict, corpus):
    # Prepare container for tweet embeddings
    inputs_ = torch.zeros((len(corpus), 100))

    # Counter for debugging purposes
    count_not_found = 0.
    total_count = 0.

    # We loop over all the tweets in the corpus
    for idx, sentence in enumerate(corpus):
        sentence_length = len(sentence)
        mean_embedding = torch.zeros(100)
        for word in sentence:
            total_count += 1
            if word in emb_dict.keys():
                mean_embedding += torch.Tensor(emb_dict[word])
            else:
                count_not_found += 1

        # We average the word embedding over the sentence
        mean_embedding /= sentence_length

        # We add the embedded sentence to the inputs tensor
        inputs_[idx] = mean_embedding
    ratio = (count_not_found / total_count) * 100

    print("Percentage of not recognised words (those we do not have an embedding for) : %.2f" % ratio, "%")
    # We return the embedded corpus
    return inputs_


#  <-------------- END Tweet preparation for NN -------------> #

#  <---------- Data Loader ------------>  #

def data_loader(emb_corpus, labels, batch_size, random_seed, valid_size=0.1, test_size=0.1, balancing=True):

    # One hot encoding of labels, function in utils
    labels_ = one_hot_encoding(np.array(labels))

    # Create dataset
    dataset_ = torch.utils.data.TensorDataset(emb_corpus, labels_)

    # Split to train / valid / test dataset
    size_dataset = len(labels)
    indices = list(range(size_dataset))
    valid_split = int(np.floor(valid_size * size_dataset))
    test_split = int(np.floor(test_size * size_dataset))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_valid_idx, test_idx = indices[test_split:], indices[:test_split]
    train_idx, valid_idx = train_valid_idx[valid_split:], train_valid_idx[:valid_split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    valid_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=test_sampler)

    # We balancing set to False, we just create a data loader from the train indices
    if not balancing:
        train_loader = torch.utils.data.DataLoader(
            dataset_, batch_size=batch_size, sampler=train_sampler)

        return train_loader, valid_loader, test_loader

    # If there is balancing to do, we first extract the training samples according to the
    # predefined indices, before using a weighted sampler
    if balancing:
        train_loader_unbalanced = torch.utils.data.DataLoader(
            dataset_, batch_size=len(labels), sampler=train_sampler)

        # Get back training data from sampler
        training_data, training_labels = next(iter(train_loader_unbalanced))
        # We get back classes from one hot encoding
        training_labels_int = training_labels.argmax(dim=1, keepdim=True)
        train_dataset = torch.utils.data.TensorDataset(training_data, training_labels)

        # WeightedSampler takes the list of weights as input
        class_sample_count = np.array(
            [len(np.where(training_labels_int == t)[0]) for t in np.unique(training_labels_int)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in training_labels_int])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        balance_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader_balanced = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, sampler=balance_sampler)

        return train_loader_balanced, valid_loader, test_loader

#  <----------- END Data loader --------------->

#  <----------- Global variables for the NN and the training --------------->

# we will train for N epochs (N times the model will see all the data)
epochs = 200

#  <----------- END Global variables for the NN and the training --------------->

emb_corpus = embed_corpus(emb_dict, tokenized_corpus)

train_loader, valid_loader, test_loader = data_loader(emb_corpus, label1, 32, 1)

# Instantiate the model
model = FFNN()

# we use the stochastic gradient descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=0.5)


for epoch in range(1, epochs + 1):
    loss_history = []
    acc_history = []
    for batch_idx, (embedding, target) in enumerate(train_loader):

        model.train()

        # we zero the gradients as they are not removed automatically
        optimizer.zero_grad()

        # squeeze is needed as the predictions are initially size (batch size, 1) and we need to remove the dimension of size 1
        predictions = model(embedding).squeeze(1)
        loss = nn.BCELoss()(predictions, target)
        loss_history.append(float(loss))
        acc_history.append(accuracy(predictions, target))

        # calculate the gradient of each parameter
        loss.backward()

        # update the parameters using the gradients and optimizer algorithm
        optimizer.step()

    epoch_loss = np.array(loss_history).mean()
    epoch_acc = np.array(acc_history).mean()

    val_acc = eval(model, valid_loader)
    print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')
    print("Valid accuracy :", val_acc)

# <------ Test Data -------> #
accuracy_test = eval(model, test_loader)
print("Accuracy on test dataset : %.2f" % accuracy_test, "%")


#######################################
# <------ SECOND MODEL : CNN -------> #
#######################################

def embed_corpus_2(emb_dict, corpus):

    tweet_lengths = [len(tweet) for tweet in corpus]
    max_len = np.max(np.array(tweet_lengths))

    # Prepare container for tweet embeddings
    inputs_ = torch.zeros((len(corpus), max_len, 100))

    # Counter for debugging purposes
    count_not_found = 0.
    total_count = 0.

    # We loop over all the tweets in the corpus
    for idx, tweet in enumerate(corpus):
        # and over all the words in a tweet
        for idx2, word in enumerate(tweet):
            total_count += 1
            if word in emb_dict.keys():
                inputs_[idx, idx2] = torch.Tensor(emb_dict[word])
            else:
                count_not_found += 1
    ratio = (count_not_found / total_count) * 100

    print("Percentage of not recognised words (those we do not have an embedding for) : %.2f" % ratio, "%")
    # We return the embedded corpus
    return inputs_



#  <----------- Global variables for the NN and the training --------------->
epochs=20

EMBEDDING_DIM = 100
OUTPUT_DIM = 2

#the hyperparameters specific to CNN

# we define the number of filters
N_OUT_CHANNELS = 100

# we define the window size
WINDOW_SIZE = 1

# we apply the dropout with the probability 0.5
DROPOUT = 0.5

#  <----------- END Global variables for the NN and the training --------------->

#  <----------- Training of the CNN --------------->


model = CNN(EMBEDDING_DIM, N_OUT_CHANNELS, WINDOW_SIZE, OUTPUT_DIM, DROPOUT)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

emb_corpus = embed_corpus_2(emb_dict, tokenized_corpus)
train_loader, valid_loader, test_loader = data_loader(emb_corpus, label1, 32, 1)

for epoch in range(1, epochs + 1):
    loss_history = []
    acc_history = []
    for batch_idx, (embedding, target) in enumerate(train_loader):

        model.train()

        # we zero the gradients as they are not removed automatically
        optimizer.zero_grad()

        # squeeze is needed as the predictions are initially size (batch size, 1) and we need to remove the dimension of size 1
        predictions = model(embedding).squeeze(1)
        loss = nn.BCELoss()(predictions, target)
        loss_history.append(float(loss))
        acc_history.append(accuracy(predictions, target))

        # calculate the gradient of each parameter
        loss.backward()

        # update the parameters using the gradients and optimizer algorithm
        optimizer.step()

    epoch_loss = np.array(loss_history).mean()
    epoch_acc = np.array(acc_history).mean()

    val_acc = eval(model, valid_loader)
    print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')
    print("Valid accuracy :", val_acc)

#  <----------- END Training of the CNN --------------->

# <------ Test Data -------> #
accuracy_test = eval(model, test_loader)
print("Accuracy on test dataset : %.2f" % accuracy_test, "%")


#######################################
# <------ THIRD MODEL : LSTM -------> #
#######################################

def eval_lstm(model, dataloader, batch_size):
    correct = 0.
    size = 0.
    for batch_idx, (embedding, target) in enumerate(dataloader):
        if embedding.shape[0] != batch_size:
            print("Batch size not common, discarding it")
            continue
        embedding = embedding.to(device)
        target = target.to(device)
        size += len(target)
        prediction = model(embedding).squeeze(1)
        correct += accuracy(prediction, target) * len(target)
    acc = float(correct / size) * 100
    return acc



batch_size = 32
epochs=10

EMBEDDING_DIM = 100
HIDDEN_DIM = 10
OUTPUT_DIM = 2
DROPOUT = 0.5
SEQ_LEN = 105

model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, SEQ_LEN).to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

emb_corpus = embed_corpus_2(emb_dict, tokenized_corpus)
train_loader, valid_loader, test_loader = data_loader(emb_corpus, label1, batch_size, 1)

for epoch in range(1, epochs + 1):
    loss_history = []
    acc_history = []
    for batch_idx, (embedding, target) in enumerate(train_loader):
        model.train()

        # we zero the gradients as they are not removed automatically
        optimizer.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Send input data to GPU
        embedding = embedding.to(device)

        # squeeze is needed as the predictions are initially size (batch size, 1) and we need to remove the dimension of size 1
        # Have to transpose batch and sequence dimensions for nn.LSTM
        # embedding = torch.transpose(embedding, 0, 1)
        predictions = model(embedding).squeeze(1)
        loss = nn.BCELoss()(predictions, target.to(device))
        loss_history.append(float(loss))
        acc_history.append(accuracy(predictions, target.to(device)))

        # calculate the gradient of each parameter
        loss.backward()

        # update the parameters using the gradients and optimizer algorithm
        optimizer.step()

    epoch_loss = np.array(loss_history).mean()
    epoch_acc = np.array(acc_history).mean()
    print("End of epoch")
    val_acc = eval_lstm(model, valid_loader, batch_size)
    print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')
    print("Valid accuracy :", val_acc)

accuracy_test = eval_lstm(model, test_loader, batch_size)
print("Accuracy on test dataset : %.2f" % accuracy_test, "%")


##################################################
# <------ FOURTH MODEL : bi-LSTM + Conv -------> #
##################################################

batch_size = 32
epochs=10

EMBEDDING_DIM = 100
HIDDEN_DIM = 10
OUTPUT_DIM = 2
DROPOUT = 0.5
SEQ_LEN = 105
CHANNELS = 16
WINDOW_SIZE = 1

model = BiLSTMConv(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT,
                   SEQ_LEN, CHANNELS, WINDOW_SIZE).to(device)

optimizer = optim.Adam(model.parameters())
loss_fn = nn.BCELoss()

emb_corpus = embed_corpus_2(emb_dict, tokenized_corpus)
train_loader, valid_loader, test_loader = data_loader(emb_corpus, label1, batch_size, 1)

for epoch in range(1, epochs + 1):
    loss_history = []
    acc_history = []
    for batch_idx, (embedding, target) in enumerate(train_loader):
        model.train()

        # we zero the gradients as they are not removed automatically
        optimizer.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Send input data to GPU
        embedding = embedding.to(device)

        # squeeze is needed as the predictions are initially size (batch size, 1) and we need to remove the dimension of size 1
        # Have to transpose batch and sequence dimensions for nn.LSTM
        # embedding = torch.transpose(embedding, 0, 1)
        predictions = model(embedding).squeeze(1)
        loss = nn.BCELoss()(predictions, target.to(device))
        loss_history.append(float(loss))
        acc_history.append(accuracy(predictions, target.to(device)))

        # calculate the gradient of each parameter
        loss.backward()

        # update the parameters using the gradients and optimizer algorithm
        optimizer.step()

    epoch_loss = np.array(loss_history).mean()
    epoch_acc = np.array(acc_history).mean()
    print("End of epoch")
    val_acc = eval_lstm(model, valid_loader, batch_size)
    print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')
    print("Valid accuracy :", val_acc)

accuracy_test = eval_lstm(model, test_loader, batch_size)
print("Accuracy on test dataset : %.2f" % accuracy_test, "%")