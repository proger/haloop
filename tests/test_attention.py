import torch
import pytest

from ha.transformer import attend, attend_chunked


@pytest.mark.parametrize("seed", range(10))
def test_eq_attend_and_chunked(seed):
    torch.manual_seed(seed)

    T, S = 10, 21
    q = torch.randn(2, 3, T, 7, device='cuda', dtype=torch.float16)
    k = torch.randn(2, 3, S, 7, device='cuda', dtype=torch.float16)    
    v = torch.randn(2, 3, S, 7, device='cuda', dtype=torch.float16)
    causal_mask = torch.triu(q.new_ones(T, S), diagonal=1).bool()

    chunked = attend_chunked(q, k, v, causal_mask, chunk_size=2)[0]
    print('chunked', chunked[0])
    full = attend(q, k, v, causal_mask)[0]
    print('full', full[0])

    assert torch.allclose(
        chunked,
        full,
        atol=2e-3,
    )
