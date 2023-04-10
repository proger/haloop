
import torch

@torch.jit.script
def transducer_forward_score1(
    transcription_probabilities, # (T, K)  # f   # time starts at 0
    prediction_probabilities, # (U, K)     # g   # first symbol is blank (0)
    targets # (U,)                         # y   # first symbol is blank (0)
):
    """Transducer forward score for a single sequence (using probabilities).

    [Graves12] Sequence Transduction with Recurrent Neural Networks
    """
    T, K = transcription_probabilities.shape
    U, K = prediction_probabilities.shape

    alpha = transcription_probabilities.new_zeros((T+1, U))
    y_prob = transcription_probabilities.new_zeros((T, U))
    blank_prob = transcription_probabilities.new_zeros((T+1, U))

    alpha[-1, 0] = 1
    blank_prob[-1, 0] = 1

    #print(alpha.T)

    for t in range(T):
        f = transcription_probabilities[t]

        for u in range(U):
            g = prediction_probabilities[u]

            h = (f + g).softmax(dim=-1)

            blank_prob[t, u] = h[0]
            y_prob[t, u] = h[targets[u]]

            alpha[t, u] = alpha[t-1, u].clone() * blank_prob[t-1, u].clone() + alpha[t, u-1].clone() * y_prob[t, u-1].clone()

        #print(alpha.T)

    return alpha[T-1, U-1] * y_prob[T-1, U-1]


if __name__ == '__main__':
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