
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

    N = 3
    inputs = 100*torch.randn(N, 100, 1, device='cuda', dtype=torch.float16)
    targets = torch.randint(0, 5, (N, 10), device='cuda', dtype=torch.long)
    input_lengths = torch.tensor([100]*N, device='cuda', dtype=torch.long)
    target_lengths = torch.tensor([10]*N, device='cuda', dtype=torch.long)

    serial_grad_norms = torch.zeros(N, device='cuda', dtype=torch.float16)
    serial_losses = torch.zeros(N, device='cuda', dtype=torch.float16)
    for i in range(N):
        system.zero_grad()
        loss = system(inputs[i:i+1], targets[i:i+1], input_lengths[i:i+1], target_lengths[i:i+1])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), 10000, foreach=False)
        print(i, 'grad_norm,loss', grad_norm.item(), loss.item(), sep='\t')
        serial_grad_norms[i] = grad_norm
        serial_losses[i] = loss

    parallel_grad_norms, parallel_losses = gradient_norms(system, inputs, targets, input_lengths, target_lengths)
    print('parallel_grad_norms', parallel_grad_norms)
    print('parallel_losses', parallel_losses)

    assert torch.allclose(
        parallel_grad_norms,
        serial_grad_norms,
        atol=1e-3,
    )

    assert torch.allclose(
        parallel_losses,
        serial_losses,
    )