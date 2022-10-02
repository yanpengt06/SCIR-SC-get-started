from typing import Tuple

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from tutils import sequence_mask


class WordEncoder(nn.Module):
    """

    input: packed_sents(not a PackedSequence object), packed_words_per_sent
    output: sents: rep of sentences, and word_alpha: word att distribution
    """

    def __init__(self,
                 vocab_size: int,
                 emb_size: int,
                 word_rnn_size: int,
                 word_att_size: int,
                 dropout: float
                 ):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.word_rnn = nn.GRU(input_size=emb_size, hidden_size=word_rnn_size, batch_first=True, bidirectional=True)
        self.h2u = nn.Linear(2 * word_rnn_size, word_att_size)
        self.u_w = nn.Linear(word_att_size, 1, bias=False)  # inner dot with u_w

    def forward(self, sents: torch.Tensor, words_per_sent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        @param sents: packed_sents
        @param words_per_sent: packed_words_per_sents
        @return:
        """
        sents = self.dropout(self.embedding(sents))  # n_sents x max_word_len -> N x W x E (emb_size)
        packed_words = pack_padded_sequence(sents, lengths=words_per_sent.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_words, _ = self.word_rnn(packed_words)
        sents, len_unpacked = pad_packed_sequence(packed_words,
                                                  batch_first=True)  # n_words x 2H -> n_sents x max_word_len x 2H
        u_it = self.h2u(sents)  # n_sents x W x 2H -> n_sents x W x U
        u_it = F.tanh(u_it)
        att_scores = self.u_w(u_it).squeeze(2)  # n_sents x W x U -> n_sents x W
        att_scores = sequence_mask(att_scores, len_unpacked, value=float("-inf"))
        word_alpha = F.softmax(att_scores, dim=1)
        sents = word_alpha.unsqueeze(2) * sents
        sents = sents.sum(dim=1)
        return sents, word_alpha


if __name__ == '__main__':
    word_encoder = WordEncoder(10, 3, 4, 5, 0)
    sents = torch.LongTensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]])
    words_per_sent = torch.LongTensor([3, 2])
    sents, word_alpha = word_encoder(sents, words_per_sent)
    print(sents)
    print(word_alpha)
