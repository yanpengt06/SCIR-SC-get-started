import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F
from word_encoder import WordEncoder
from tutils import sequence_mask


class LSTM(nn.Module):
    def __init__(self, emb_dim, class_num, vocab_size, hidden_size):
        super(LSTM, self).__init__()
        # EmbeddingLayer
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, class_num)

    def forward(self, sents: torch.Tensor, len_per_sent: torch.Tensor) -> torch.Tensor:
        """

        @param sents: B x token_limit
        @param len_per_sent: B
        @return: B x class_num -> logits
        """
        embeddings = self.embeddings(sents)  # B x M x E
        # packed_seq
        packed_ebd = pack_padded_sequence(embeddings, len_per_sent.tolist(), batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.rnn(packed_ebd)  # ht: 1 x B x H
        ht = ht.squeeze(0)
        logits = self.fc(ht)  # B x C
        return logits


class TextCNN(nn.Module):
    def __init__(self, kernel_num, emb_dim, dropout, class_num, vocab_size, pad_id):
        super(TextCNN, self).__init__()
        self.embbedings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.conv1_1 = nn.Conv2d(1, kernel_num, (3, emb_dim))
        self.conv1_2 = nn.Conv2d(1, kernel_num, (4, emb_dim))
        self.conv1_3 = nn.Conv2d(1, kernel_num, (5, emb_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * kernel_num, class_num)

    def forward(self, X, len_per_sent):
        """

        @param len_per_sent:
        @param X: sents padded   dim: B x M
        @return: logits B x C
        """
        X = self.embbedings(X)  # B x M x E
        X = X.unsqueeze(1)
        o_conv1_1 = F.relu(self.conv1_1(X))  # B x output_channels x (M - k_size + 1) x input_channels
        o_conv1_2 = F.relu(self.conv1_2(X))  # B x output_channels x (M - k_size + 1) x input_channels
        o_conv1_3 = F.relu(self.conv1_3(X))  # B x output_channels x (M - k_size + 1) x input_channels
        o_pool_1_1 = F.max_pool1d(o_conv1_1.squeeze(-1), o_conv1_1.shape[2])  # B x output_channels x 1
        o_pool_1_2 = F.max_pool1d(o_conv1_2.squeeze(-1), o_conv1_2.shape[2])
        o_pool_1_3 = F.max_pool1d(o_conv1_3.squeeze(-1), o_conv1_3.shape[2])
        cat = self.dropout(torch.cat([o_pool_1_1, o_pool_1_2, o_pool_1_3], dim=1).squeeze(-1))  # B x 300
        output = self.fc(cat)  # B x C
        return output


class HAN(nn.Module):
    """
        reproduction on original paper
        input docs: B x S x W
        output logits: B x C
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sent_rnn_size, class_num, word_att_size, sent_att_size,
                 dropout=0.5):
        super(HAN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.word_rnn_size = word_rnn_size
        self.sent_rnn_size = sent_rnn_size
        self.class_num = class_num
        self.word_att_size = word_att_size
        self.sent_att_size = sent_att_size
        self.word_encoder = WordEncoder(vocab_size, emb_size, word_rnn_size, word_att_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.w_s = nn.Linear(2 * sent_rnn_size, sent_att_size)
        self.sentence_rnn = nn.GRU(input_size=2 * word_rnn_size, hidden_size=sent_rnn_size, batch_first=True,
                                   bidirectional=True)
        self.u_s = nn.Linear(sent_att_size, 1, bias=False)  # U_s
        self.fc = nn.Linear(2 * sent_rnn_size, class_num)

    def forward(self, docs, sents_per_doc, words_per_sent):
        """

        @param docs:
        @param sents_per_doc:
        @param words_per_sent:
        @return: logits: B x C
        """
        packed_sents = pack_padded_sequence(docs, lengths=sents_per_doc.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_words_per_sent = pack_padded_sequence(words_per_sent, lengths=sents_per_doc.to('cpu'), batch_first=True,
                                                     enforce_sorted=False)
        sents, word_alpha = self.word_encoder(packed_sents.data,
                                              packed_words_per_sent.data)  # get the word att-distribution
        sents = self.dropout(sents)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(
            data=sents,
            batch_sizes=packed_sents.batch_sizes,
            sorted_indices=packed_sents.sorted_indices,
            unsorted_indices=packed_sents.unsorted_indices))

        docs, unpacked_len = pad_packed_sequence(packed_sentences, batch_first=True)  # B x max_sent_len x 2H_sent
        u_it = self.w_s(docs)  # B x max_sent_len x 2H_sent -> B x max_sent_len x U
        u_it = F.tanh(u_it)
        att_scores = self.u_s(u_it).squeeze(2)  # B x max_sent_len
        att_scores = sequence_mask(att_scores, unpacked_len, value=float("-inf"))
        sent_alpha = F.softmax(att_scores, dim=1)
        docs = (sent_alpha.unsqueeze(2) * docs)  # B x max_sent_len x 2H
        docs = docs.sum(dim=1)  # B x 2H

        word_alpha, _ = pad_packed_sequence(PackedSequence(
            data = word_alpha,
            batch_sizes = packed_sentences.batch_sizes,
            sorted_indices = packed_sentences.sorted_indices,
            unsorted_indices = packed_sentences.unsorted_indices
        ), batch_first = True)

        logits = self.fc(self.dropout(docs)) # B x 2H_sent -> B x C

        return logits, sent_alpha, word_alpha


if __name__ == '__main__':
    pass
    han = HAN(38000, 3, 5, 10, 5, 6, 7, 0)
    docs = torch.LongTensor([[[1, 2, 3, 37426, 4, 5, 6, 3, 7, 8,
                               9, 10, 5, 11, 3, 12, 13, 8, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [14, 15, 16, 17, 6, 18, 19, 20, 21, 3,
                               9, 10, 22, 8, 23, 17, 37426, 24, 25, 5,
                               26, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [27, 28, 6, 18, 29, 30, 31, 32, 5, 3,
                               33, 34, 3, 35, 5, 36, 18, 37, 5, 3,
                               38, 39, 3, 40, 41, 42, 43, 8, 3, 44,
                               45, 46, 47, 37426, 3, 48, 45, 46, 49, 0],
                              [23, 18, 50, 51, 5, 52, 53, 3, 54, 8,
                               55, 3, 22, 52, 56, 5, 36, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [57, 8, 58, 3, 52, 15, 59, 23, 45, 60,
                               8, 3, 61, 37426, 62, 37426, 5, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [63, 64, 52, 5, 65, 66, 67, 68, 3, 69,
                               70, 71, 52, 72, 23, 73, 13, 5, 21, 3,
                               12, 74, 8, 3, 75, 76, 77, 78, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                             [[79, 80, 9, 10, 81, 11, 5, 82, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [83, 84, 22, 4, 85, 86, 87, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [88, 89, 90, 91, 92, 3, 93, 11, 94, 5, 95, 96, 97, 98,
                               3, 99, 100, 101, 102, 80, 11, 103, 5, 104, 3, 22, 105, 66,
                               94, 8, 106, 107, 108, 3, 109, 110, 66, 94, 8, 111],
                              [85, 112, 83, 113, 114, 115, 116, 117, 118, 5, 3, 119, 120, 95,
                               121, 70, 122, 123, 5, 3, 124, 125, 126, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [127, 128, 3, 129, 114, 27, 130, 131, 132, 3, 119, 23, 27, 9,
                               117, 83, 84, 5, 133, 134, 3, 135, 136, 137, 3, 138, 139, 33,
                               140, 141, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])
    sents_per_doc = torch.LongTensor([6, 5])
    words_per_sent = torch.LongTensor([[18, 21, 39, 17, 17, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0],
                                       [8, 7, 40, 23, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0]])
    output, sent_alpha, word_alpha = han(docs, sents_per_doc, words_per_sent)
    print(word_alpha)
