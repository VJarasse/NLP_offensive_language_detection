import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class tweetsDataset(Dataset):
    def __init__(self, corpus, label1,, label2, label3, transform=None):
        """
        Args:
            corpus (list): list of cleaned sentences from the corpus
            label1, label2, label3 (lists) : list of tweets labels transformed to integers
                for the 3 subtasks

        """
        self.corpus = corpus
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sample = self.corpus[idx]
        label1 = self.label1[idx]
        label2 = self.label2[idx]
        label3 = self.label3[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label1, label2, label3
