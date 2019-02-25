import torch
from tqdm import tqdm
import codecs
import random


def accuracy(output, target):
    output = output.argmax(dim = 1, keepdim=True)
    target = target.argmax(dim = 1, keepdim=True)
    correct = (output == target).float()
    acc = float(correct.sum( ) /len(output) )
    return acc


def eval(model, dataloader):
    correct = 0.
    size = 0.
    for batch_idx, (embedding, target) in enumerate(dataloader):
        size += len(target)
        prediction = model(embedding).squeeze(1)
        correct += accuracy(prediction, target) * len(target)
    acc = float(correct / size) * 100
    return acc


def get_word2idx(vocabulary):
    word2idx = {w: idx + 1 for (idx, w) in enumerate(vocabulary)}
    # we reserve the 0 index for the placeholder token
    word2idx['<pad>'] = 0
    return word2idx


def tokenize(corpus, sep=' '):
    tokenized_corpus = []
    for sentence in corpus:
        tokenized_sentence = []
        for token in sentence.split(sep):
            tokenized_sentence += token
        tokenized_corpus.append(tokenized_sentence)
    return tokenized_corpus


def get_model_inputs(tokenized_corpus, word2idx, labels, max_len):
    # we index our sentences
    vectorized_sents = [[word2idx[tok] for tok in sent if tok in word2idx] for sent in tokenized_corpus]

    # we create a tensor of a fixed size filled with zeroes for padding

    sent_tensor = torch.zeros((len(vectorized_sents), max_len)).long()

    sent_lengths = [len(sent) for sent in vectorized_sents]

    # we fill it with our vectorized sentences

    for idx, (sent, sentlen) in enumerate(zip(vectorized_sents, sent_lengths)):
        sent_tensor[idx, :sentlen] = torch.LongTensor(sent)

    label_tensor = torch.FloatTensor(labels)

    return sent_tensor, label_tensor