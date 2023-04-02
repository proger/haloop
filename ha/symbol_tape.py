import torch
from pathlib import Path


class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1

    def state_dict(self):
        return {
            'id_to_string': self.id_to_string,
            'pad_id': self.pad_id,
            'unk_id': self.unk_id,
        }

    def load_state_dict(self, state_dict):
        self.id_to_string = state_dict['id_to_string']
        self.string_to_id = {v: k for k, v in self.id_to_string.items()}
        self.pad_id = state_dict['pad_id']
        self.unk_id = state_dict['unk_id']

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


def tokenize_bytes(text_file, vocab, extend_vocab=True, device='cpu'):
    if vocab is None:
        vocab = Vocabulary(pad_token=0, unk_token=7) # nul and bel

    full_text = []
    print(f"Reading binary file from: {text_file}")
    with open(text_file, 'rb') as bin:
        for line in bin:
            for byte in line:
                full_text.append(vocab.get_idx(byte, extend_vocab=extend_vocab))

    data = torch.tensor(full_text, device=device, dtype=torch.int16)
    return data, vocab


def tokenize_chars(text_file, vocab, extend_vocab=True, device='cpu'):
    if vocab is None:
        vocab = Vocabulary()

    full_text = []
    print(f"Reading text file from: {text_file}")
    with open(text_file, 'r') as text:
        for line in text:
            tokens = list(line)
            for token in tokens:
                full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

    data = torch.tensor(full_text, device=device, dtype=torch.int16)
    return data, vocab



class SymbolTape:
    def __init__(self, data, batch_size, bptt_len, pad_id):
        self.batches = self.create_batch(data, batch_size, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, batch_size, bptt_len, pad_id):
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // batch_size + 1

        padded = input_data.data.new_full((segment_len * batch_size,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(batch_size, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, batch_size), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches


