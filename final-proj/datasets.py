from typing import Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

"""
    this file contains various datasets.
"""


class ReviewDataset(Dataset):
    """
        encode and pad to token_limit
    """

    def __init__(self, sents, labels, word_map, token_limit):
        self.word_map = word_map
        self.sents = sents
        self.labels = labels
        self.token_limit = token_limit

    def __getitem__(self, i: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """encode and pad"""

        enc_sent = list(map(lambda word: self.word_map.get(word, self.word_map['<unk>']), self.sents[i]))
        enc_pad_sent = enc_sent + [0] * (200 - len(enc_sent))

        # test set doen't have labels
        if np.isnan(self.labels[i]):
            return torch.LongTensor(enc_pad_sent), torch.LongTensor([-5]), torch.LongTensor(
                [len(enc_sent)])
        # label shift : [-2,-1,0,1] -> [0,1,2,3]
        return torch.LongTensor(enc_pad_sent), torch.LongTensor([self.labels[i] + 2]), torch.LongTensor([len(enc_sent)])

    def __len__(self) -> int:
        return len(self.sents)


class DocDataset(Dataset):
    """

    """
    def __init__(self, docs, labels, word_map, sent_limit, word_limit):
        self.docs = docs
        self.labels = labels
        self.word_map = word_map
        self.word_limit = word_limit
        self.sent_limit = sent_limit

    def __getitem__(self, idx: str) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        encode and pad
        @param idx:
        @return:
        """
        doc = self.docs[idx]
        word_padded = list(map(lambda s:
            list(map(lambda w: self.word_map.get(w, self.word_map["<unk>"]) ,s)) + [0] * (self.word_limit - len(s))
            ,doc))
        sent_padded = word_padded + [[0] * self.word_limit] * (self.sent_limit - len(word_padded))
        sents_per_doc = len(doc)
        words_per_sent =  [len(s) for s in doc] + [0] * (self.sent_limit - sents_per_doc)

        # shift label to [0,1,2,3]
        return  torch.LongTensor(sent_padded), torch.LongTensor([self.labels[idx] + 2]), torch.LongTensor([sents_per_doc]),\
                torch.LongTensor(words_per_sent)

    def __len__(self):
        return len(self.docs)