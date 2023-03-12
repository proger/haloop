
import torch

def ctc_forward_score1(
    emissions, # (T, C)
    targets, # (N,), such that T > N
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
    targets, # (B, N,), such that T > N
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
        same_symbols = _t_a_r_g_e_t_s_[2:] == _t_a_r_g_e_t_s_[:-2]
        transitions = torch.where(
            same_symbols,
            self_loop.logaddexp(prev_symbol),
            transitions
        )

        if t > 1:
            # first symbol past t=1 only comes from a self loop
            self_loop_ = log_alpha[t-1, 1:]
            log_alpha[t, 1:] = self_loop_ + emissions[t, _t_a_r_g_e_t_s_[1:]]

        log_alpha[t, 2:] = transitions + emissions[t, _t_a_r_g_e_t_s_[2:]]

    return -log_alpha[T-1, -1].logaddexp(log_alpha[T-1, -2])


def ctc_forward_score3(emissions, targets):
    emissions = emissions.permute(1, 0, 2) # (B, T, C)
    return torch.vmap(ctc_forward_score2)(emissions, targets)



if __name__ == '__main__':
    torch.manual_seed(2)
    logits = torch.randn(5, 7).log_softmax(-1)
    print('logits')
    print(logits.T)
    targets = torch.LongTensor([1,2,3])
    print('scores')
    print(ctc_forward_score1(logits, targets))

    print(ctc_forward_score2(logits, targets))

    print(torch.nn.functional.ctc_loss(logits[:, :],
                                       targets,
                                       torch.full((1,), 5),
                                       torch.full((1,), 3), blank=0, reduction='none'))

    print(ctc_forward_score3(logits[:, None, :], targets[None, :]))