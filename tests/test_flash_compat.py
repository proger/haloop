import pytest
import torch

from flash_attn.modules.mha import MHA

from ha.transformer import MultiHeadAttention


@pytest.fixture
def input_tensor():
    torch.manual_seed(0)
    return torch.randn(1, 7, 1024, dtype=torch.float16).cuda()


def test_mha_with_flash_attn(input_tensor):
    mha = MHA(
        embed_dim=16*64,
        num_heads=16,
        cross_attn=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        causal=False,
        rotary_emb_dim=0,
        use_flash_attn=True
    ).cuda().to(torch.float16)

    att = MultiHeadAttention(
        head_dim=64,
        heads=16,
        p_drop=False
    ).cuda().to(torch.float16).init_from_flash_mha_(mha)

    assert torch.allclose(
        att(input_tensor, input_tensor, measure_entropy=True, apply_rotations=False)[0],
        mha(input_tensor),
        atol=1e-3
    )


def test_mha_with_rotary_emb(input_tensor):
    mha = MHA(
        embed_dim=16*64,
        num_heads=16,
        cross_attn=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        causal=False,
        rotary_emb_dim=64,
        rotary_emb_interleaved=True, # XXX
        use_flash_attn=True
    ).cuda().to(torch.float16)

    att = MultiHeadAttention(
        head_dim=64,
        heads=16,
        p_drop=False
    ).cuda().to(torch.float16).init_from_flash_mha_(mha)

    assert torch.allclose(
        att(input_tensor, input_tensor, measure_entropy=True, rope=True)[0],
        mha(input_tensor),
        atol=1e-3
    )
