
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
    _t_a_r_g_e_t_s_ = torch.cat([_t_a_r_g_e_t_s_, targets.new(1).fill_(blank)], dim=-1)

    log_alpha = emissions.new_full((T, len(_t_a_r_g_e_t_s_)), float('-inf'))

    log_alpha[0, 0] = emissions[0, _t_a_r_g_e_t_s_[0]]
    log_alpha[0, 1] = emissions[0, _t_a_r_g_e_t_s_[1]]

    for t in range(1, T):
        for s in range(1, len(_t_a_r_g_e_t_s_)):
            log_alpha_bar = log_alpha[t-1, s].logaddexp(log_alpha[t-1, s-1])
            if _t_a_r_g_e_t_s_[s] == 0 or _t_a_r_g_e_t_s_[s-2] == _t_a_r_g_e_t_s_[s]:
                pass
            else:
                log_alpha_bar = log_alpha_bar.logaddexp(log_alpha[t-1, s-2])
            log_alpha[t, s] = log_alpha_bar + emissions[t, _t_a_r_g_e_t_s_[s]]

    return -log_alpha[T-1, -1].logaddexp(log_alpha[T-1, -2])



if __name__ == '__main__':
    torch.manual_seed(2)
    logits = torch.randn(5, 7).log_softmax(-1)
    print(logits.T)
    targets = torch.LongTensor([1,2,3])
    print(ctc_forward_score1(logits, targets))

    print(torch.nn.functional.ctc_loss(logits[:, :],
                                       targets,
                                       torch.full((1,), 5),
                                       torch.full((1,), 3), blank=0, reduction='none'))