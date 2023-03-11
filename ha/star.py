import torch


def add_stars_to_targets(
    log_probs, # (T, V)
    targets, # (T)
    penalty=-5,
):
    """
    For a label sequence A B C, produce a regex-like sequence [^A]+ A [^B]+ B [^C]+ C .*
    
    [Pratap22] Star Temporal Classification: Sequence Classification with Partially Labeled Data
               Vineel Pratap, Awni Hannun, Gabriel Synnaeve, Ronan Collobert
               https://arxiv.org/abs/2201.12208
    """
    T, V = log_probs.shape

    # Make one symbol <star> that has a probability of a sum of all symbols except a blank at position V.
    # For each vocabulary entry t except a blank (0), at position V+t,
    # make a new symbol <star>\t that has a probability of a sum (logadd) of all other symbols except t.

    complete_star_log_probs = -log_probs[:, 1:].logsumexp(dim=1, keepdim=True)
    
    star_log_probs = log_probs.new_zeros(T, V + V)
    star_log_probs[:, :V] = log_probs
    star_log_probs[:, V] = complete_star_log_probs.squeeze() + penalty
    star_log_probs[:, V+1:] = complete_star_log_probs.logaddexp(log_probs[:, 1:]) + penalty

    lse = star_log_probs.logsumexp(dim=1, keepdim=True)
    star_log_probs -= lse # renormalize

    # Make a new target string of size 2*T + 1 where each symbol t is preceded by <star>\t.
    # The last symbol is <star>.

    star_targets = (V + targets).repeat_interleave(2)
    star_targets[::2] = targets    
    star_targets = torch.cat([star_targets, star_targets.new([V])])

    return star_log_probs, star_targets


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)

    targets = torch.tensor([1,2,3]) # in real situation, the number of frames must be larger than the doubled number of targets
    V = 55
    probs = (torch.nn.functional.one_hot(targets, num_classes=V).float() + 0.0001) / (1 + 0.0001 * V)
    logits = torch.log(probs)
    print(logits.T)
    print(logits.T.logsumexp(dim=0, keepdim=True))
    
    star_logits, star_targets = add_stars_to_targets(logits, targets)
    
    print(star_logits.T, star_targets)
    lse = star_logits.T.logsumexp(dim=0, keepdim=True)
    print(lse)
    #print(star_logits.T - lse)
    print((star_logits.T - lse).logsumexp(dim=0, keepdim=True)) # must be zeros