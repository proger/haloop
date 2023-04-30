import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctc import ctc_forward_score3, ctc_reduce_mean
from .star import star_ctc_forward_score

class Encoder(nn.Module):
    def __init__(self, input_dim=13, subsample_dim=128, hidden_dim=1024):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.subsample = nn.Conv1d(input_dim, subsample_dim, 5, stride=4, padding=3)
        #self.lstm = nn.LSTM(subsample_dim, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(subsample_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.2)

    def subsampled_lengths(self, input_lengths):
        # https://github.com/vdumoulin/conv_arithmetic
        p, k, s = self.subsample.padding[0], self.subsample.kernel_size[0], self.subsample.stride[0]
        o = input_lengths + 2 * p - k
        o = torch.floor(o / s + 1)
        return o.int()

    def forward(self, inputs):
        x = inputs
        x = self.subsample(x.mT).mT
        x = x.relu()
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return x.relu()



class CTCRecognizer(nn.Module):
    def __init__(self, feat_dim=1024, vocab_size=55+1):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0)
        self.classifier = nn.Linear(feat_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def log_probs(self, features):
        features = self.dropout(features)
        features = self.classifier(features)
        return features.log_softmax(dim=-1)

    def forward(self, features, targets, input_lengths=None, target_lengths=None):
        if input_lengths is None:
            input_lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.long)
        if target_lengths is None:
            target_lengths = torch.full((features.shape[0],), len(targets), dtype=torch.long)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.log_probs(features).to(torch.float32)
            logits = logits.permute(1, 0, 2) # T, N, C
            loss = self.ctc(logits, targets, input_lengths=input_lengths, target_lengths=target_lengths)
            #loss = ctc_reduce_mean(ctc_forward_score3(logits, targets, input_lengths, target_lengths), target_lengths)
            return loss


class StarRecognizer(nn.Module):
    def __init__(self, feat_dim=1024, vocab_size=55+1, star_penalty=-1):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.star_penalty = star_penalty

    def log_probs(self, features):
        features = self.dropout(features)
        features = self.classifier(features)
        return features.log_softmax(dim=-1)

    def forward(self, features, targets, input_lengths=None, target_lengths=None):
        if input_lengths is None:
            input_lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.long)
        if target_lengths is None:
            target_lengths = torch.full((features.shape[0],), len(targets), dtype=torch.long)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.log_probs(features).to(torch.float32)

            logits = logits.permute(1, 0, 2) # T, N, C
            losses = star_ctc_forward_score(logits, targets, input_lengths, target_lengths,
                                            star_penalty=self.star_penalty)
            loss = ctc_reduce_mean(losses, target_lengths)
            return loss


if __name__ == '__main__':
    from .xen import Vocabulary
    encoder = Encoder()
    reco = CTCRecognizer()
    vocabulary = Vocabulary()
    x = torch.randn(1, 13, 320).mT
    x = encoder(x)
    print(x.shape)
