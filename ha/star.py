import torch


def logsubexp(b, a):
    return b + torch.log1p(-torch.exp(a - b))



def add_stars_to_targets(
    log_probs, # (N, T, V)
    targets, # (N, T)
    penalty=-10,
):
    """
    For a label sequence A B C, produce a regex-like sequence [^A]+ A [^B]+ B [^C]+ C .*

    Assumes blank has position 0.

    Output targets are of size T' = 2*T + 1.
    
    [Pratap22] Star Temporal Classification: Sequence Classification with Partially Labeled Data
               Vineel Pratap, Awni Hannun, Gabriel Synnaeve, Ronan Collobert
               https://arxiv.org/abs/2201.12208
    """
    N, T, V = log_probs.shape

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

    # Make a new target string of size 2*T + 1 where each symbol t is preceded by <star>\t.
    # The last symbol is <star>.

    star_targets = torch.stack([V + targets, targets], dim=1).mT.reshape(N, -1)
    star_targets = torch.cat([star_targets, targets.new(N, 1).fill_(V)], dim=-1)

    return star_log_probs, star_targets


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)

    targets = torch.tensor([1,2,3]) # in real situation, the number of frames must be larger than the doubled number of targets
    V = 55
    probs = (torch.nn.functional.one_hot(targets, num_classes=V).float() + 0.0001) / (1 + 0.0001 * V)
    logits = torch.log(probs)
    print(logits.T)
    print(logits.T.logsumexp(dim=0, keepdim=True))
    
    logits = logits[None, :] # (N, T, V)
    targets = targets[None, :] # (N, T)

    star_logits, star_targets = add_stars_to_targets(logits, targets)

    star_logits = star_logits.squeeze(0) # (T, V+V)
    star_targets = star_targets.squeeze(0) # (2*T+1)

    print(star_logits.T, star_targets)
    lse = star_logits.T.logsumexp(dim=0, keepdim=True)
    print(lse)
    #print(star_logits.T - lse)
    print((star_logits.T - lse).logsumexp(dim=0, keepdim=True)) # must be zeros

    # test batching:

    logits = logits.repeat(3, 1, 1) # (N, T, V)
    targets = targets.repeat(3, 1) # (N, T)
    add_stars_to_targets(logits, targets)