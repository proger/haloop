import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=13, subsample_dim=128, hidden_dim=1024):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.subsample = nn.Conv1d(input_dim, subsample_dim, kernel_size=5, stride=4, padding=3)
        #self.lstm = nn.LSTM(subsample_dim, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(subsample_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.2)

    def subsampled_lengths(self, input_lengths):
        # https://github.com/vdumoulin/conv_arithmetic
        p, k, s = self.subsample.padding[0], self.subsample.kernel_size[0], self.subsample.stride[0]
        o = input_lengths + 2 * p - k
        o = torch.floor(o / s + 1)
        return o.int()

    def forward(self, inputs, input_lengths):
        x = inputs
        x = self.subsample(x.mT).mT
        x = x.relu()
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return x.relu()



class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout=0.0):
        super().__init__()

        self.num_classes = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

        self.out_layer.weight = self.embedding.weight

    def forward(self,
                input, # (T, N)
                state # ((L, N, H), (L, N, H))
                ):
        emb = self.embedding(input)

        output, state = self.rnn(emb, state)
        output = self.out_layer(output) # (T, N, V)

        output = output.view(-1, self.num_classes)
        return output, state

    def forward_batch_first(self,
                            input, # (N, T),
                            state # ((L, N, H), (L, N, H))
                            ):
        emb = self.embedding(input)
        emb = emb.transpose(0,1) # (T, N, H)

        output, state = self.rnn(emb, state)
        output = self.out_layer(output) # (T, N, V)
        output = output.transpose(0,1) # (N, T, V)

        return output, state

    def init_hidden(self, batch_size=1):
        weight = self.out_layer.weight
        h = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        c = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h,c)

    def truncate_hidden(self, state):
        h,c = state
        return (h.detach(), c.detach())

