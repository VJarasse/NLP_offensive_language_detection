import torch


def accuracy(output, target):
    output = torch.round(torch.sigmoid(output))
    correct = (output == target).float()
    acc = correct.sum( ) /len(correct)
    return acc


def get_word2idx(vocabulary):
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocabulary)}
    # we reserve the 0 index for the placeholder token
    word2idx['<pad>'] = 0
    return word2idx