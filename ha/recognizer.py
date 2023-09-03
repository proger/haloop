import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol

from .ctc import ctc_forward_score3, ctc_reduce_mean
from .star import star_ctc_forward_score
from .transducer import transducer_forward_score
from .rnn import Decoder


class Decodable(Protocol):
    def log_probs(self, features):
        ...

    def decode(
        self,
        features: torch.Tensor, # (N, T, C)
        input_lengths: torch.LongTensor | None = None, # (N,)
        target_lengths: torch.LongTensor | None = None, # (N,)
    ) -> tuple[torch.Tensor, torch.Tensor, list, torch.Tensor, torch.Tensor]: # sequences, lengths, alignments, scores, sum_entropies
        ...

    def forward(
        self,
        features: torch.Tensor, # (N, T, C)
        targets: torch.Tensor, # (N, S)
        input_lengths: torch.LongTensor | None = None, # (N,)
        target_lengths: torch.LongTensor | None = None, # (N,)
        star_penalty: float | None = None,
        measure_entropy: bool = False,
        drop_labels : bool = False
    ) -> tuple[torch.Tensor, dict]:
        ...


class TemporalClassifier(nn.Module, Decodable):
    def __init__(self, feat_dim=1024, vocab_size=256):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def log_probs(self, features):
        features = self.dropout(features)
        features = self.classifier(features)
        return features.log_softmax(dim=-1)

    def decode(self, features, input_lengths, target_lengths):
        logits = self.log_probs(features)

        scores, alignments = logits.max(dim=-1) # maybe use input_lengths?
        hypotheses = torch.nested.nested_tensor([
            [i for i in torch.unique_consecutive(alignment) if i]
            for alignment in alignments # greedy
        ])
        output_lengths = torch.tensor([len(hypothesis) for hypothesis in hypotheses])

        #decoded_seqs, _decoded_logits = ctc_beam_search_decode_logits(seq) # FIXME: speed it up
        return hypotheses, output_lengths, alignments, scores, None

    def forward(self, features, targets, input_lengths=None, target_lengths=None, star_penalty=None, measure_entropy=False):
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
                return loss, {}
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                logits = self.log_probs(features).to(torch.float32)

                logits = logits.permute(1, 0, 2) # T, N, C
                losses = star_ctc_forward_score(logits, targets, input_lengths, target_lengths,
                                                star_penalty=self.star_penalty)
                loss = ctc_reduce_mean(losses, target_lengths)
                return loss, {}


class Transducer(nn.Module, Decodable):
    def __init__(self, feat_dim=1024, vocab_size=256):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, vocab_size)
        self.lm = Decoder(vocab_size, emb_dim=512, hidden_dim=512, num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.2)

    def decode(self, features, input_lengths):
        raise NotImplementedError()

    def forward(
        self,
        features,
        targets,
        input_lengths=None,
        target_lengths=None,
        star_penalty=None # ignored
    ):
        N, _, _ = features.shape
        hidden = self.lm.init_hidden(N)

        # input needs to start with 0
        lm_targets = torch.cat([targets.new_zeros((N, 1)), targets], dim=1) # (N, U1)

        lm_outputs, _ = self.lm.forward_batch_first(lm_targets, hidden) # (N, U1, C)

        features = self.dropout(features)
        features = self.classifier(features) # (N, T, C)

        joint = features[:, :, None, :] + lm_outputs[:, None, :, :] # (N, T, U1, C)

        if False:
            joint = joint.log_softmax(dim=-1)
            losses = transducer_forward_score(joint, targets, input_lengths, target_lengths)
            loss = ctc_reduce_mean(losses, target_lengths)
        else:
            from torchaudio.functional import rnnt_loss
            loss = rnnt_loss(joint,
                                targets.to(torch.int32),
                                input_lengths.to(torch.int32),
                                target_lengths.to(torch.int32),
                                blank=0, reduction='mean', fused_log_softmax=True)
        return loss, {}


if __name__ == '__main__':
    from .xen import Vocabulary
    from .rnn import Encoder
    encoder = Encoder()
    reco = TemporalClassifier()
    vocabulary = Vocabulary()
    x = torch.randn(1, 13, 320).mT
    x = encoder(x)
    print(x.shape)
