import torch

from ha.transformer import AudioEncoder
from ha.attention_audio import StridingAudioEncoder
from ha.init import StridingAudioEncoderConfig


def load_from_striding(self: AudioEncoder, state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith('conv.'):
            new_state_dict['conv.' + key] = state_dict[key]
        elif key.startswith('transformer.'):
            k = key.replace('transformer.', '')
            if 'Wqkv' in key:
                _, i, _ = k.split('.', maxsplit=2)
                step = state_dict[key].shape[0] // 3
                new_state_dict[f'h.{i}.mix_time.q.weight'] = state_dict[key][0*step:1*step, :]
                new_state_dict[f'h.{i}.mix_time.k.weight'] = state_dict[key][1*step:2*step, :]
                new_state_dict[f'h.{i}.mix_time.v.weight'] = state_dict[key][2*step:3*step, :]
            elif 'out_proj' in key:
                _, i, _ = k.split('.', maxsplit=2)
                new_state_dict[f'h.{i}.mix_time.proj.weight'] = state_dict[key]
            elif 'rotary_emb' in key:
                pass # ignore those
            else:
                new_state_dict[
                    k.replace(
                        'mlp.c_fc', 'mix_chan.0'
                    ).replace(
                        'mlp.c_proj', 'mix_chan.2',
                    ).replace(
                        'ln_1', 'ln_time'
                    ).replace(
                        'ln_2', 'ln_chan',
                    )
                ] = state_dict[key]

    self.load_state_dict(new_state_dict, strict=True)


def test_eq_blocks():
    torch.manual_seed(0)
    audio_encoder = AudioEncoder(
        head_dim=64,
        heads=4,
        p_drop=0.2,
        layers=2,
        input_dim=80,
        conv_dim=256,
        conv_strides=(2,2,2),
    ).cuda().to(torch.float16).eval()
    config = StridingAudioEncoderConfig(
        block_size=-1,
        vocab_size=-1,
        n_layer=2,
        n_head=4,
        n_embd=64*4,
        dropout=0.2,
        d_conv=256,
        conv_strides=(2,2,2),
    )
    striding_audio_encoder = StridingAudioEncoder(config).cuda().to(torch.float16).eval()
    load_from_striding(audio_encoder, striding_audio_encoder.state_dict())

    N, T, C = 1, 1000, 80
    inputs = torch.randn(N, T, C).cuda().to(torch.float16)
    input_lengths = torch.tensor([T])

    y1 = striding_audio_encoder(inputs, input_lengths)[0]
    y2 = audio_encoder(inputs, input_lengths)[0]
    assert torch.allclose(
        y1, y2, atol=1e-2
    ), f'\n{y1}\n!=\n{y2}'
