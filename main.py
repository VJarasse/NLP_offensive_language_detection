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
from torch.utils.data.sampler import SubsetRandomSampler

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
for line in glove:v
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

#  <---------- Data Loader ------------>  #


def data_loader(emb_dict, corpus, labels, batch_size, random_seed, valid_size=0.1, test_size=0.1, balancing=True):
    inputs_ = torch.zeros((len(corpus), 100))

    # Counter for debbugging purposes
    count_not_found = 0.
    total_count = 0.

    for idx, sentence in enumerate(corpus):
        sentence_length = len(sentence),
        mean_embedding = torch.zeros(100)
        for word in sentence:
            total_count += 1
            if word in emb_dict.keys():
                mean_embedding += torch.Tensor(emb_dict[word])
            else:
                # print("Word not found in dictionnary")
                count_not_found += 1

        # We average the word embedding over the sentence
        mean_embedding /= torch.Tensor(sentence_length).view(-1)

        # We add the emmebdded sentence to the inputs tensor
        inputs_[idx] = mean_embedding

    # One hot encoding of labels
    labels_ = torch.zeros(len(corpus), 2)
    for idx, label in enumerate(labels):
        labels_[idx, int(label)] = 1.

    if balancing:
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dataset_ = torch.utils.data.TensorDataset(inputs_, labels_)
    '''
    # Split to train / valid / test dataset
    size_dataset = len(corpus)
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

    train_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=batch_size, sampler=test_sampler)

    return train_loader, valid_loader, test_loader, count_not_found / total_count
    '''
    train_loader = DataLoader(dataset_, batch_size=batch_size, num_workers=1, sampler=sampler)

    return train_loader
#  <----------- END Data loader --------------->


#  <----------- Global variables for the NN and the training --------------->

# we will train for N epochs (N times the model will see all the data)
epochs = 3


#  <----------- END Global variables for the NN and the training --------------->

# Create dataloaders

#train_sent_tensor, train_label_tensor = get_model_inputs(tokenized_corpus, word2idx, label1, max_len)
train_loader = data_loader(emb_dict, tokenized_corpus, label1, batch_size=32, random_seed=1)

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

    val_acc = eval(model, train_loader)
    print(f'| Epoch: {epoch:02} | Train Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc*100:.2f}%')
    print("Valid accuracy :", val_acc)