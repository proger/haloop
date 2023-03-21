import torch


def logsubexp(b, a):
    return b + torch.log1p(-torch.exp(a - b))


def intersperse_stars(
    log_probs, # (T, N, V)
    targets, # (N, S), such that S < T
    penalty=0,
):
    """
    For a label sequence A B C, produce a regex-like sequence [^A]+ A [^B]+ B [^C]+ C .*

    Assumes blank has position 0.

    Output targets are of size S_ = 2*S + 1.
    
    [Pratap22] Star Temporal Classification: Sequence Classification with Partially Labeled Data
               Vineel Pratap, Awni Hannun, Gabriel Synnaeve, Ronan Collobert
               https://arxiv.org/abs/2201.12208
    """
    T, N, V = log_probs.shape

    # Make one symbol <star> that has a probability of a sum of all symbols except a blank at position V.
    # For each vocabulary entry t except a blank (0), at position V+t,
    # make a new symbol <star>\t that has a probability of a sum (logadd) of all other symbols except t.

    complete_star_log_probs = log_probs[:, :, 1:].logsumexp(dim=-1, keepdim=True)

    allstar_log_probs = complete_star_log_probs + penalty
    starsub_log_probs = logsubexp(complete_star_log_probs, log_probs[:, :, 1:]) + penalty
    star_log_probs = torch.cat([
        log_probs,
        allstar_log_probs,
        starsub_log_probs
    ], dim=-1)

    # lse = star_log_probs.logsumexp(dim=-1, keepdim=True)
    # star_log_probs = star_log_probs - lse # renormalize

    # Make a new target string of size 2*S + 1 where each symbol t is preceded by <star>\t.
    # The last symbol is <star>.

    star_targets = torch.stack([V + targets, targets], dim=1).mT.reshape(N, -1)
    star_targets = torch.cat([star_targets, targets.new(N, 1).fill_(V)], dim=-1)

    return star_log_probs, star_targets # (T, N, 2*V), (N, 2*S+1)


def intersperse_blanks(
    targets, # (N, S)
    blank=0
):
    N, S = targets.shape

    # A B C -> _ A _ B _ C _
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(targets, blank), targets], dim=1).mT.reshape(N, -1)
    _t_a_r_g_e_t_s_ = torch.cat([_t_a_r_g_e_t_s_, targets.new_full((N, 1), blank)], dim=-1)

    return _t_a_r_g_e_t_s_  # (N, 2*S+1)


def star_ctc_forward_score(
    emissions, # (T, N, C)
    targets, # (N, S,), such that T > S
    emission_lengths, # (N,)
    target_lengths, # (N,)
    star_penalty=-0.5,
    animate=False
):
    """
    CTC forward score with stars for a batch of sequences.

    [Graves06] Connectionist Temporal Classification:
               Labelling Unsegmented Sequence Data with Recurrent Neural Networks
               Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber

    [Pratap22] Star Temporal Classification: Sequence Classification with Partially Labeled Data
               Vineel Pratap, Awni Hannun, Gabriel Synnaeve, Ronan Collobert
               https://arxiv.org/abs/2201.12208
    """

    blank = 0
    T, N, C = emissions.shape
    N, S = targets.shape

    emissions, targets = intersperse_stars(emissions, targets)
    V = 2*C
    assert emissions.shape[-1] == V
    targets = intersperse_blanks(targets, blank=blank)  # (N, 4*S+3)

    if animate:
        void = float('-inf') # animation looks nicer with -inf, but gradients are NaN
    else:
        void = torch.finfo(torch.float).min # float('-inf')


    S_ = targets.shape[1] # S_ = 4*S + 3

    s_pad_top = 4 # pad with 4 states at the top of a log_alpha trellis
    s_pad_bottom = 1 # pad with 1 state at the bottom of a log_alpha trellis
    t_pad = 1 # pad with 1 time step in the beginning of a log_alpha trellis
    log_alpha = emissions.new_full((T+t_pad, N, S_+s_pad_top+s_pad_bottom), void)
    log_alpha[0, :, :s_pad_top] = 0.
    s = 4 # start at the 4th state

    T_last = t_pad + emission_lengths - 1
    S_last = s_pad_top + 4*target_lengths + 3 - 1

    log_alpha[:, :, -s_pad_bottom] = -7007.7007 # toot-toot

    Ss_ = torch.arange(S_, device=emissions.device) # (S_,)
    blanks = Ss_ % 2 == 0
    stars = Ss_ % 4 == 1
    same_label_as_prev = (targets[:, 4+3::4] == targets[:, 3:-4:4]).repeat_interleave(4, dim=-1)
    same_label_as_prev = torch.cat([same_label_as_prev.new_zeros(N, 4), same_label_as_prev, same_label_as_prev.new_zeros(N, 3)], dim=-1)

    for t in range(t_pad, T+t_pad):
        prev = log_alpha[t-1].clone()

        from_prev_label = prev[:, s-4:-5]
        from_first_blank = prev[:, s-3:-4]
        from_star = prev[:, s-2:-3]
        from_star_blank = prev[:, s+1:]
        from_prev = prev[:, s-1:-2]
        from_self = prev[:, s:-1]

        # Look at transitions into these kinds of symbols:
        # into_blank: any blank before star_t, blank before label or final blank
        # into_star: any star
        # into_label: any known label
        # into_same_label: any known label that is the same as the previous label

        from_prev_or_self = from_prev.logaddexp(from_self)
        from_first_blank_or_prev_or_star = from_first_blank.logaddexp(from_prev).logaddexp(from_star)
        into_blank = from_prev_or_self
        into_star = from_prev_or_self.logaddexp(from_star_blank) + star_penalty
        into_different_label = from_first_blank_or_prev_or_star.logaddexp(from_prev_label)
        into_same_label = from_first_blank_or_prev_or_star

        transitions = torch.where(blanks, into_blank,
                                  torch.where(stars, into_star,
                                              torch.where(same_label_as_prev, into_same_label, into_different_label)))

        # emissions are shifted by 1
        log_alpha[t, :, s:-1] = transitions + emissions[t-1].gather(-1, targets)

        if animate:
            print(log_alpha[:, 1, :].T)
            import time; time.sleep(0.5)

    #print('S_last', S_last, log_alpha.shape, sep='\n')

    Ns = torch.arange(N, device=emissions.device)
    last_timestep = log_alpha[T_last, Ns, :]
    last_blank  = last_timestep[Ns, S_last]
    last_star = last_timestep[Ns, S_last-1]
    last_blank_1  = last_timestep[Ns, S_last-2]
    last_symbol  = last_timestep[Ns, S_last-3]

    return -last_blank.logaddexp(last_star).logaddexp(last_blank_1).logaddexp(last_symbol)




def test_intersperse():
    torch.set_printoptions(precision=4, sci_mode=False)

    targets = torch.tensor([1,2,3]) # in real situation, the number of frames must be larger than the doubled number of targets
    V = 55
    probs = (torch.nn.functional.one_hot(targets, num_classes=V).float() + 0.0001) / (1 + 0.0001 * V)
    logits = torch.log(probs)
    print(logits.T)
    print(logits.T.logsumexp(dim=0, keepdim=True))
    
    logits = logits[:, None, :] # (T, N, V)
    targets = targets[None, :] # (N, T)

    star_logits, star_targets = intersperse_stars(logits, targets)

    star_logits = star_logits.squeeze(1) # (T, V+V)
    star_targets = star_targets.squeeze(0) # (2*T+1)

    print(star_logits.T, star_targets)
    lse = star_logits.T.logsumexp(dim=0, keepdim=True)
    print(lse)
    #print(star_logits.T - lse)
    print((star_logits.T - lse).logsumexp(dim=0, keepdim=True)) # must be zeros

    # test batching:

    logits = logits.repeat(3, 1, 1) # (N, T, V)
    targets = targets.repeat(3, 1) # (N, T)
    intersperse_stars(logits, targets)


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)

    torch.manual_seed(2)
    logits0 = torch.randn(10, 7).log_softmax(-1)
    logits1 = torch.randn(10, 7).log_softmax(-1)
    targets0 = torch.LongTensor([1,2,3,3])
    targets1 = torch.LongTensor([1,2,3,4])
    input_lengths = torch.LongTensor([5, 10])
    target_lengths = torch.LongTensor([3, 4])
    logits = torch.stack([logits0, logits1], dim=1)
    print('logits.shape', logits.shape)
    targets = torch.stack([targets0, targets1], dim=0)

    print(logits, targets)

    star_ctc_forward_score(logits, targets, input_lengths, target_lengths, star_penalty=-100, animate=True)
