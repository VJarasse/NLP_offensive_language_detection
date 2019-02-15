import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FFNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_len, num_classes):

        super(FFNN, self).__init__()

        # embedding (lookup layer) layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # activation
        self.relu1 = nn.ReLU()

        # output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        embedded = self.embedding(x)
        # we average the embeddings of words in a sentence
        averaged = embedded.mean(1)
        # (batch size, max sent length, embedding dim) to (batch size, embedding dim)
        out = self.fc1(averaged)
        out = self.relu1(out)
        out = self.fc2(out)
        return out