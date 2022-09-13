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
