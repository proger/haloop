import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctc import ctc_forward_score3, ctc_reduce_mean
from .star import star_ctc_forward_score


class Recognizer(nn.Module):
    def __init__(self, feat_dim=1024, vocab_size=256):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def log_probs(self, features):
        features = self.dropout(features)
        features = self.classifier(features)
        return features.log_softmax(dim=-1)

    def decode(self, features, input_lengths):
        logits = self.log_probs(features)

        alignments = logits.argmax(dim=-1)
        hypotheses = torch.nested.nested_tensor([
            [i for i in torch.unique_consecutive(alignment) if i]
            for alignment in alignments # greedy
        ])

        #decoded_seqs, _decoded_logits = ctc_beam_search_decode_logits(seq) # FIXME: speed it up
        return hypotheses, alignments, input_lengths

    def forward(self, features, targets, input_lengths=None, target_lengths=None, star_penalty=None):
        if input_lengths is None:
            input_lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.long)
        if target_lengths is None:
            target_lengths = torch.full((features.shape[0],), len(targets), dtype=torch.long)

        if star_penalty is None:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                logits = self.log_probs(features)
                logits1 = logits.to(torch.float32).permute(1, 0, 2) # T, N, C
                loss = F.ctc_loss(logits1, targets, input_lengths=input_lengths, target_lengths=target_lengths)
                #loss = ctc_reduce_mean(ctc_forward_score3(logits, targets, input_lengths, target_lengths), target_lengths)
                return loss
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                logits = self.log_probs(features).to(torch.float32)

                logits = logits.permute(1, 0, 2) # T, N, C
                losses = star_ctc_forward_score(logits, targets, input_lengths, target_lengths,
                                                star_penalty=self.star_penalty)
                loss = ctc_reduce_mean(losses, target_lengths)
                return loss


if __name__ == '__main__':
    from .xen import Vocabulary
    from .rnn import Encoder
    encoder = Encoder()
    reco = Recognizer()
    vocabulary = Vocabulary()
    x = torch.randn(1, 13, 320).mT
    x = encoder(x)
    print(x.shape)
