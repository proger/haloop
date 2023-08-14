
import torch

from ha.transformer import CTCAttentionDecoder, AudioEncoder
from ha.active import MiniSystem, gradient_norms


def test_gradient_norms():
    """
    Computing gradient norms per sample using vmap can be done in parallel,
    clip_grad_norm_ can compute the norm of the whole batch of a sample.
    They are the same when the batch size is 1.
    """
    torch.manual_seed(0)
    encoder = AudioEncoder(layers=1, head_dim=2, heads=1, input_dim=1, conv_dim=2, conv_strides=(1, 1))
    decoder = CTCAttentionDecoder(vocab=6, head_dim=encoder.head_dim, heads=encoder.heads, p_drop=0.1, layers=1)

    encoder = encoder.to('cuda').to(torch.float16)
    decoder = decoder.to('cuda').to(torch.float16)
    system = MiniSystem(encoder, decoder).eval()

    x = torch.randn(3, 100, 1, device='cuda', dtype=torch.float16)
    y = torch.randint(0, 5, (3, 10), device='cuda', dtype=torch.long)
    y_kbest = y[None, ...]
    input_lengths = torch.tensor([100]*3, device='cuda', dtype=torch.long)
    target_lengths = torch.tensor([10]*3, device='cuda', dtype=torch.long)
    target_kbest_lengths = target_lengths[None, ...]

    K,N,U = y_kbest.shape
    system.zero_grad()
    system(
        x[:1],
        y_kbest[0,:1],
        input_lengths[:1],
        target_kbest_lengths[0,:1]
    ).backward()
    # print('lm+head', system.decoder.lm_head.weight.grad)
    # #print(x, y_kbest, input_lengths, target_kbest_lengths)
    # print(x.shape, y_kbest.shape, input_lengths.shape, target_kbest_lengths.shape)
    # print(x[:1].shape, y_kbest[:,:1].shape, input_lengths[:1].shape, target_kbest_lengths[:,:1].shape)
    assert torch.allclose(
        gradient_norms(system, x[:1], y_kbest[:,:1], input_lengths[:1], target_kbest_lengths[:,:1]),
        torch.nn.utils.clip_grad_norm_(system.parameters(), 10000, foreach=False),
    )
