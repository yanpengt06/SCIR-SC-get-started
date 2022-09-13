import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        embeddings = self.embeddings(sents) # B x M x E
        # packed_seq
        packed_ebd = pack_padded_sequence(embeddings, len_per_sent.tolist(), batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.rnn(packed_ebd) # ht: 1 x B x H
        ht = ht.squeeze(0)
        logits = self.fc(ht) # B x C
        return logits


if __name__ == '__main__':
    lstm = LSTM(1,2,4,3)
    batch = torch.LongTensor([[3,2,1,1,0],[1,2,3,0,0]])
    logits = lstm(batch, torch.Tensor([4,3]))
    print(logits)




