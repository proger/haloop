
import torch


def transducer_forward_score1(
    transcription_probabilities, # (T, K)  # f   # time starts at 1 (zero-padded by one frame)
    prediction_probabilities, # (U, K)     # g   # first symbol is blank (0)
    targets # (U,)                         # y   # first symbol is blank (0)
):
    """Transducer forward score for a single sequence (using probabilities).

    [Graves12] Sequence Transduction with Recurrent Neural Networks
    """
    T, K = transcription_probabilities.shape
    U, K = prediction_probabilities.shape

    alpha = transcription_probabilities.new_zeros((T, U))
    y_prob = transcription_probabilities.new_zeros((T, U))
    blank_prob = transcription_probabilities.new_zeros((T, U))

    alpha[0, 0] = 1
    blank_prob[0, 0] = 1

    for t in range(1, T):
        for u in range(U):
            f = transcription_probabilities[t]
            g = prediction_probabilities[u]

            h = (f + g).softmax(dim=-1)

            blank_prob[t, u] = h[0]
            y_prob[t, u] = h[targets[u]]

            alpha[t, u] = alpha[t-1, u] * blank_prob[t-1, u] + alpha[t, u-1] * y_prob[t, u-1]

    return alpha[t, u] * y_prob[t, u]


if __name__ == '__main__':
    transcription_probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.1, 0.2, 0.3, 0.4],
                                                [0.1, 0.2, 0.3, 0.4],
                                                [0.1, 0.2, 0.3, 0.4]])
    prediction_probabilities = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                             [0.1, 0.2, 0.3, 0.4],
                                             [0.1, 0.2, 0.3, 0.4]])
    targets = torch.tensor([0, 1, 2])

    print(transducer_forward_score1(transcription_probabilities, prediction_probabilities, targets))