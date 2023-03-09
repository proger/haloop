
import torch


def ctc_beam_search_decode_probs(
    emit_probs, # T K
    beam_size=3,
):
    """
    CTC Beam Search with probabilities, assumes blank is 0

    [Graves14] Towards End-To-End Speech Recognition with Recurrent Neural Networks
               https://proceedings.mlr.press/v32/graves14.html
    """
    blank = 0
    time_steps, vocab_size = emit_probs.shape # T, K

    # we start with a single empty sequence from a blank transition with probability 1
    top_seqs = [[]]
    seq_probs = torch.ones(1, dtype=torch.float32, device=emit_probs.device)
    blank_probs = torch.ones(1, dtype=torch.float32, device=emit_probs.device)
    label_probs = torch.zeros(1, dtype=torch.float32, device=emit_probs.device)

    for t in range(time_steps):
        seqs = top_seqs[:] # proposed sequences with no changes go first

        ext_label_probs = torch.zeros(len(top_seqs), vocab_size, dtype=torch.float32, device=emit_probs.device)

        for s, seq in enumerate(top_seqs):
            # case adding nothing via same-label transition
            if seq:
                label_probs[s] *= emit_probs[t, seq[-1]]

                try:
                    prefix_index = top_seqs.index(seq[:-1])
                except ValueError:
                    pass
                else:
                    label_probs[s] += emit_probs[t, seq[-1]] * 1 * blank_probs[prefix_index]

            # case adding nothing via a blank transition
            blank_probs[s] = seq_probs[s] * emit_probs[t, blank]

            # case adding one label via other-label transition
            last_sym = seq[-1] if seq else blank
            last_sym = torch.nn.functional.one_hot(torch.tensor(last_sym, device=device), num_classes=vocab_size).float()

            trans_prob = 1. # LM probability that this transition is valid

            ext_label_probs[s, :] = last_sym * blank_probs[s] + (1-last_sym) * seq_probs[s]
            ext_label_probs[s, :] = emit_probs[t] * trans_prob * ext_label_probs[s, :]

            # now, we are extending seqs K times with K different symbols added
            seqs.extend([seq + [k] for k in range(0, vocab_size)])

        blank_probs = torch.cat((blank_probs, torch.zeros(len(top_seqs) * vocab_size, dtype=torch.float32, device=emit_probs.device)), dim=0)
        label_probs = torch.cat((label_probs, ext_label_probs.view(-1)), dim=0)
        seq_probs = blank_probs + label_probs

        topk = seq_probs.topk(beam_size, dim=0, largest=True, sorted=True)
        seq_probs = topk.values
        blank_probs = blank_probs[topk.indices]
        label_probs = label_probs[topk.indices]

        top_seqs = [seqs[i] for i in topk.indices]


    return top_seqs, seq_probs


def ctc_beam_search_decode_logits(
    emit_logits, # T K
    beam_size=3,
    dtype=torch.float32,
):
    """
    CTC Beam Search with log probabilities, assumes blank is 0

    [Graves14] Towards End-To-End Speech Recognition with Recurrent Neural Networks
               https://proceedings.mlr.press/v32/graves14.html
    """
    blank = 0
    time_steps, vocab_size = emit_logits.shape # T, K
    device = emit_logits.device

    # we start with a single empty sequence from a blank transition with probability 1
    top_seqs = [[]]
    seq_logits = torch.zeros(1, dtype=dtype, device=device)
    blank_logits = torch.zeros(1, dtype=dtype, device=device)
    label_logits = torch.empty(1, dtype=dtype, device=device).fill_(float('-inf'))

    for t in range(time_steps):
        seqs = top_seqs[:] # proposed sequences with no changes go first

        ext_label_logits = torch.zeros(len(top_seqs), vocab_size, dtype=dtype, device=device)

        for s, seq in enumerate(top_seqs):
            # case adding nothing via same-label transition
            if seq:
                label_logits[s] += emit_logits[t, seq[-1]]

                try:
                    prefix_index = top_seqs.index(seq[:-1])
                except ValueError:
                    pass
                else:
                    label_logits[s] = torch.logaddexp(label_logits[s], emit_logits[t, seq[-1]] + 0 + blank_logits[prefix_index])

            # case adding nothing via a blank transition
            blank_logits[s] = seq_logits[s] + emit_logits[t, blank]

            # case adding one label via other-label transition
            last_sym = seq[-1] if seq else blank
            last_sym = torch.nn.functional.one_hot(torch.tensor(last_sym, device=device), num_classes=vocab_size)

            trans_prob = 0. # LM log probability that this transition is valid

            ext_label_logits[s, :] = torch.where(last_sym.bool(), blank_logits[s], seq_logits[s])
            ext_label_logits[s, :] = emit_logits[t] + trans_prob + ext_label_logits[s, :]
            #print(t, 'ext_label_logits2', ext_label_logits)

            # now, we are extending seqs K times with K different symbols added
            seqs.extend([seq + [k] for k in range(0, vocab_size)])

        blank_logits = torch.cat((blank_logits, torch.zeros(len(top_seqs) * vocab_size, dtype=dtype, device=device)), dim=0)
        label_logits = torch.cat((label_logits, ext_label_logits.view(-1)), dim=0)
        seq_logits = torch.logaddexp(blank_logits, label_logits)

        topk = seq_logits.topk(beam_size, dim=0, largest=True, sorted=True)
        seq_logits = topk.values
        blank_logits = blank_logits[topk.indices]
        label_logits = label_logits[topk.indices]

        top_seqs = [seqs[i] for i in topk.indices]


    return top_seqs, seq_logits


if __name__ == '__main__':
    #logits = torch.nn.functional.one_hot(torch.tensor([0,3,1,2])).float()
    #logits = torch.nn.functional.one_hot(torch.tensor([4,3,1,2])).float()
    #logits = torch.nn.functional.one_hot(torch.tensor([4,4,4,4])).float()
    probs = torch.nn.functional.one_hot(torch.tensor([0,3,1,2,2,0,0,2,0,0,0,1,2,3])).float()
    print('probs', probs.shape, '\n', probs.T) # transpose so that time step frames are printed as columns
    print(ctc_beam_search_decode_probs(probs))
    logits = torch.log(probs)
    print('logits', logits.shape, '\n', logits.T) # transpose so that time step frames are printed as columns
    print(ctc_beam_search_decode_logits(logits))