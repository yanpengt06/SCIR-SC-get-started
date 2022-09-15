import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


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


if __name__ == '__main__':
    pass
    # lstm = LSTM(1,2,4,3)
    # batch = torch.LongTensor([[3,2,1,1,0],[1,2,3,0,0]])
    # logits = lstm(batch, torch.Tensor([4,3]))
    # print(logits)
    # model = TextCNN(100, 3, 0, 4, 5, 0)
    # sents = torch.LongTensor([[1,2,3,0,0],[2,4,0,0,0]])
    # cnn_output = model(sents)
    # print(cnn_output.shape)
