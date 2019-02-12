import pandas as pd
import io

dataframe = pd.read_csv(io.BytesIO(uploaded['offenseval-training-v1.tsv']), sep = "\t", header = 0)

# http://pytorch.org/
from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

#@title Loading packages

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def tokenize(corpus, sep=' '):
  tokenized_corpus = []
  for sentence in corpus:
    tokenized_sentence = []
    for token in sentence.split(sep):
      tokenized_sentence.append(token)
    tokenized_corpus.append(tokenized_sentence)
  return tokenized_corpus

def extract_voc(tokenized_corpus):
  vocabulary = []
  for sentence in tokenized_corpus:
    for token in sentence:
      if token not in vocabulary:
        vocabulary.append(token)
  return vocabulary

tokenized_corpus = tokenize(dataframe['tweet'],' ')
vocabulary = extract_voc(tokenized_corpus)
print(len(vocabulary))


def clean_token(token):
    clean_token = token.lower()
    if len(clean_token) < 2:
        return clean_token

    out_char = ['.', ',', '?', "'", '!', ";"]
    chunk_1 = [clean_token]
    chunk_2 = []
    for sep in out_char:
        for chunk in chunk_1:
            chunk_2 += chunk.split(sep)
        chunk_1 = chunk_2
        chunk_2 = []
    output = []
    for element in chunk_1:
        if element != "":
            output.append(element)
    return output

def clean_vocabulary(vocabulary):
  clean_voc = []
  for element in vocabulary:
    clean_element = clean_token(element)
    for chunk in clean_element:
      if not chunk in clean_voc:
        clean_voc.append(chunk)
  return clean_voc

clean_vocabulary = clean_vocabulary(vocabulary)
vocabulary_size = len(clean_vocabulary)
print(vocabulary_size)

word2idx = {w: idx for (idx, w) in enumerate(clean_vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(clean_vocabulary)}

def look_up_table(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x