
import torch

#@torch.jit.script
def transducer_forward_score1(
    transcription_probs, # (T, K)  # f   # time starts at 0
    prediction_probs, # (U, K)     # g   # first symbol is blank (0)
    targets # (U,)                 # y   # first symbol is blank (0)
):
    """Transducer forward score for a single sequence (using probabilities).

    [Graves12] Sequence Transduction with Recurrent Neural Networks
    """
    T, K = transcription_probs.shape
    U, K = prediction_probs.shape

    joint_probs = (transcription_probs[:, None, :] + prediction_probs[None, :, :]).softmax(dim=-1)

    alpha = transcription_probs.new_zeros((T, U))
    t = 0
    alpha[t, 0] = 1
    for u in range(1, U):
        prev_symbol_prob = joint_probs[t, u-1, targets[u-1]]
        alpha[t, u] = alpha[t-1, u].clone() * 1. + alpha[t, u-1].clone() * prev_symbol_prob

    #print(t, alpha.T)

    for t in range(1, T):
        u = 0
        prev_blank_prob = joint_probs[t-1, u, 0]
        alpha[t, u] = alpha[t-1, u].clone() * prev_blank_prob

        for u in range(1, U):
            prev_blank_prob  = joint_probs[t-1, u, 0]
            prev_symbol_prob = joint_probs[t, u-1, targets[u-1]]
            alpha[t, u] = alpha[t-1, u].clone() * prev_blank_prob + alpha[t, u-1].clone() * prev_symbol_prob

        #print(t, alpha.T)

    return alpha[T-1, U-1] * joint_probs[T-1, U-1, 0]




def test_simple1():
    transcription_probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.1, 0.2, 0.3, 0.4],
                                                [0.1, 0.2, 0.3, 0.4],
                                                [0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    prediction_probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                             [0.1, 0.2, 0.3, 0.4],
                                             [0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    targets = torch.tensor([0, 1, 2])

    loss = transducer_forward_score1(transcription_probabilities, prediction_probabilities, targets)
    print(loss)
    loss.backward()
    print(transcription_probabilities.grad)


def _test_bigger_compile2():
    # does not work: data-dependent operators
    torch.manual_seed(42)

    with torch.device('cuda:1'):

        transcription_probabilities = torch.randn(400, 31, requires_grad=True)
        prediction_probabilities = torch.randn(10, 31, requires_grad=True)
        targets = torch.randint(0, 30, (10,))

        f = torch.compile(transducer_forward_score1, mode='reduce-overhead', fullgraph=True)

        for _ in range(100):
            loss = f(transcription_probabilities, prediction_probabilities, targets)
            loss.backward()


def _test_bigger_script2():
    torch.manual_seed(42)

    transcription_probabilities = torch.randn(400, 31, requires_grad=True)
    prediction_probabilities = torch.randn(10, 31, requires_grad=True)
    targets = torch.randint(0, 30, (10,))

    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        f = torch.jit.script(transducer_forward_score1)

        for _ in range(1):
            with record_function("forward+backward"):
                loss = f(transcription_probabilities, prediction_probabilities, targets)
                loss.backward()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == '__main__':
    import pytest
    pytest.main(["--no-header", "-v", "-s", __file__])
