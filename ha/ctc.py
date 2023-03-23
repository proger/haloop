
import torch

def ctc_forward_score1(
    emissions, # (T, C)
    targets, # (S,), such that T > S
):
    """
    CTC forward score for a single sequence.

    [Graves06] Connectionist Temporal Classification:
               Labelling Unsegmented Sequence Data with Recurrent Neural Networks
    """
    blank = 0
    T, C = emissions.shape

    # A B C -> _ A _ B _ C _
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(targets, blank), targets], dim=0).mT.reshape(-1)
    _t_a_r_g_e_t_s_ = torch.cat([_t_a_r_g_e_t_s_, targets.new_full((1,), blank)], dim=-1)

    log_alpha = emissions.new_full((T, len(_t_a_r_g_e_t_s_)), float('-inf'))

    log_alpha[0, :2] = emissions[0, _t_a_r_g_e_t_s_[:2]]

    for t in range(1, T):
        for s in range(1, len(_t_a_r_g_e_t_s_)):
            self_loop = log_alpha[t-1, s]
            prev_symbol = log_alpha[t-1, s-1]
            skip = log_alpha[t-1, s-2]

            base_transitions = self_loop.logaddexp(prev_symbol)
            transitions_with_skip = base_transitions.logaddexp(skip)

            # transition into blank: no skips across blanks
            transitions = torch.where(
                _t_a_r_g_e_t_s_[s] == blank,
                base_transitions,
                transitions_with_skip
            )

            # transition from the same symbol: must go through a blank (no skips)
            transitions = torch.where(
                _t_a_r_g_e_t_s_[s-2] == _t_a_r_g_e_t_s_[s],
                base_transitions,
                transitions
            )

            log_alpha[t, s] = transitions + emissions[t, _t_a_r_g_e_t_s_[s]]

    return -log_alpha[T-1, -1].logaddexp(log_alpha[T-1, -2])



def ctc_forward_score2(
    emissions, # (T, C)
    targets, # (S,), such that T > S
):
    """
    CTC forward score.

    [Graves06] Connectionist Temporal Classification:
               Labelling Unsegmented Sequence Data with Recurrent Neural Networks
    """

    blank = 0
    T, C = emissions.shape

    # A B C -> _ A _ B _ C _
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(targets, blank), targets], dim=0).mT.reshape(-1)
    _t_a_r_g_e_t_s_ = torch.cat([_t_a_r_g_e_t_s_, targets.new_full((1,), blank)], dim=-1)

    log_alpha = emissions.new_full((T, len(_t_a_r_g_e_t_s_)), float('-inf'))

    log_alpha[0, :2] = emissions[0, _t_a_r_g_e_t_s_[:2]]

    # first symbol at t=1 comes from a self loop or a leading blank
    log_alpha[1,  1:2] = log_alpha[0, 0].logaddexp(log_alpha[0, 1]) + emissions[1, _t_a_r_g_e_t_s_[1:2]]

    for t in range(1, T):
        self_loop = log_alpha[t-1, 2:]
        prev_symbol = log_alpha[t-1, 1:-1]
        skip = log_alpha[t-1, :-2]

        # transition into blank: no skips across blanks
        blanks = _t_a_r_g_e_t_s_[2:] == blank
        transitions = torch.where(
            blanks,
            self_loop.logaddexp(prev_symbol),
            self_loop.logaddexp(prev_symbol).logaddexp(skip)
        )

        # transition from the same symbol: must go through a blank (no skips)
        same_label_as_prev = (_t_a_r_g_e_t_s_[3:-1:2] == _t_a_r_g_e_t_s_[1:-3:2]).repeat_interleave(2, dim=-1)
        same_label_as_prev = torch.cat([same_label_as_prev, same_label_as_prev.new_zeros(1)], dim=-1)
        transitions = torch.where(
            same_label_as_prev,
            self_loop.logaddexp(prev_symbol),
            transitions
        )

        # first symbol past t=1 only comes from a self loop or a leading blank
        into_first = log_alpha[t-1, 1:].logaddexp(log_alpha[t-1, :-1])
        log_alpha[t, 1:] = into_first + emissions[t, _t_a_r_g_e_t_s_[1:]]

        log_alpha[t, 2:] = transitions + emissions[t, _t_a_r_g_e_t_s_[2:]]

    return -log_alpha[T-1, -1].logaddexp(log_alpha[T-1, -2])


def ctc_forward_score3(
    emissions, # (T, N, C)
    targets, # (N, S,), such that T > S
    emission_lengths, # (N,)
    target_lengths, # (N,)
):
    """
    CTC forward score for a batch of sequences.

    [Graves06] Connectionist Temporal Classification:
               Labelling Unsegmented Sequence Data with Recurrent Neural Networks
    """

    blank = 0
    T, N, C = emissions.shape

    # A B C -> _ A _ B _ C _
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(targets, blank), targets], dim=1).mT.reshape(N, -1)
    _t_a_r_g_e_t_s_ = torch.cat([_t_a_r_g_e_t_s_, targets.new_full((N, 1), blank)], dim=-1)
    S_ = _t_a_r_g_e_t_s_.shape[1] # S_ = 2*S + 1

    T_last = emission_lengths - 1
    S_last = 2*target_lengths + 1 - 1

    #log_alpha = emissions.new_full((T, N, S_), float('-inf'))
    log_alpha = emissions.new_full((T, N, S_), torch.finfo(torch.float).min)

    # initial states at t are a leading blank or a leading symbol
    log_alpha[0, :, :2] = emissions[0, :].gather(-1, _t_a_r_g_e_t_s_[:, :2])

    blanks = _t_a_r_g_e_t_s_[:, 2:] == blank
    same_label_as_prev = (_t_a_r_g_e_t_s_[:, 3:-1:2] == _t_a_r_g_e_t_s_[:, 1:-3:2]).repeat_interleave(2, dim=-1)
    same_label_as_prev = torch.cat([same_label_as_prev, same_label_as_prev.new_zeros(N, 1)], dim=-1)

    for t in range(1, T):
        prev = log_alpha[t-1].clone()

        # zero symbol comes from a self loop
        into_first_blank = prev[:, :1]
        log_alpha[t, :, :1] = into_first_blank + emissions[t].gather(-1, _t_a_r_g_e_t_s_[:, :1])

        # first symbol comes from a self loop or a leading blank
        into_first = prev[:, 1:2].logaddexp(prev[:, 0:1])
        log_alpha[t, :, 1:2] = into_first + emissions[t].gather(-1, _t_a_r_g_e_t_s_[:, 1:2])

        from_self = prev[:, 2:]
        from_prev_symbol = prev[:, 1:-1]
        from_skip = prev[:, :-2]

        from_self_or_prev_symbol = from_self.logaddexp(from_prev_symbol)
        into_blanks_or_same = from_self_or_prev_symbol
        into_labels = from_self_or_prev_symbol.logaddexp(from_skip)

        transitions = torch.where(blanks | same_label_as_prev,
                                  into_blanks_or_same,
                                  into_labels)

        log_alpha[t, :, 2:] = transitions + emissions[t].gather(-1, _t_a_r_g_e_t_s_[:, 2:])

    Ns = torch.arange(N)
    last_timestep = log_alpha[T_last, Ns, :]
    last_blank  = last_timestep[Ns, S_last]
    last_symbol = last_timestep[Ns, S_last-1]

    return -last_blank.logaddexp(last_symbol)


def ctc_reduce_mean(losses, target_lengths):
    return (losses / target_lengths).mean(-1)


if __name__ == '__main__':
    torch.manual_seed(2)
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, edgeitems=7)

    logits0 = torch.randn(5, 7).log_softmax(-1)
    logits = logits0
    print('logits')
    print(logits.T)
    targets = torch.LongTensor([1,2,3,3])
    print('scores')
    print(ctc_forward_score1(logits, targets))

    print(ctc_forward_score2(logits, targets))

    logits1 = torch.randn(5, 7).log_softmax(-1)
    targets = torch.LongTensor([1,2,3,3])
    targets1 = torch.LongTensor([1,2,3,4])
    input_lengths = torch.LongTensor([5, 5])
    target_lengths = torch.LongTensor([3, 4])
    logits = torch.stack([logits, logits1], dim=1)
    targets = torch.stack([targets, targets1], dim=0)

    print(logits, targets)

    print('ctc3     ', ctc_forward_score3(
        logits, targets,
        input_lengths,
        target_lengths))

    print('torch ctc', torch.nn.functional.ctc_loss(
        logits,
        targets,
        input_lengths,
        target_lengths, blank=0, reduction='none'
    ))

    print('ctc2[0]    ', ctc_forward_score2(logits0, torch.LongTensor([1,2,3])))
    print('ctc2[1]    ', ctc_forward_score2(logits1, targets1))


    logits2 = torch.randn(20, 7).log_softmax(-1)
    targets2 = torch.LongTensor([1,1,2,3,3,2,1,3,4,4,2,2,3])
    input_lengths = torch.LongTensor([len(logits2)])
    target_lengths = torch.LongTensor([len(targets2)])
    logits = torch.stack([logits2], dim=1)
    targets = torch.stack([targets2], dim=0)

    print('ctc3     ', ctc_reduce_mean(ctc_forward_score3(
        logits, targets,
        input_lengths,
        target_lengths), target_lengths))

    print('torch ctc', torch.nn.functional.ctc_loss(
        logits,
        targets,
        input_lengths,
        target_lengths, blank=0, reduction='mean' # ooooh
    ))
