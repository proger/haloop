import torch
from pathlib import Path
import sys
import math

from . import xen


class Vocabulary:
    def __init__(self, pad_token="·"):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # shortcut access
        self.pad_id = self.unk_id = 0

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
        return self.string_to_id[string]

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        try:
            byte = bytes([ord(string)])
            if byte in self.string_to_id:
                return self.string_to_id[byte]
        except ValueError:
            pass

        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            return self.add_new_word(string)
        else:
            return self.unk_id

    def encode(self, text, extend_vocab=False):
        try:
            return torch.LongTensor([self.get_idx(char, extend_vocab=extend_vocab) for char in text])
        except:
            import ipdb; ipdb.set_trace()

    def decode(self, ids):
        if isinstance(self.id_to_string[0], bytes):
            return b''.join([self.id_to_string[id] for id in ids])
        else:
            return ''.join([self.id_to_string[id] for id in ids])

    @classmethod
    def bytes(cls, n=256):
        self = Vocabulary(pad_token=0, unk_token=7)
        self.id_to_string = {}
        self.string_to_id = {}

        for x in range(n):
            byte = bytes([x])
            y = self.add_new_word(byte)
            assert x == y
            if x == 0: # nul
                self.pad_id = x
            elif x == 7: # bel
                self.unk_id = x

        return self

    @classmethod
    def ascii(cls):
        self = Vocabulary(pad_token=0, unk_token=7)
        self.id_to_string = {}
        self.string_to_id = {}

        for i, x in enumerate("""ε␁␂␃␄␅␆␇␈␉␤⇥␌␍␎␏␐␑␒␓␔␕␖␗␘␙␚␛␜␝␞␟ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~␡"""):
            y = self.add_new_word(x)
            assert y == i
            if i == 0: # nul
                self.pad_id = x
            elif i == 7: # bel
                self.unk_id = x

        return self

    def format(self, s):
        if isinstance(s, bytes):
            try:
                s = s.decode('utf-8')
            except UnicodeDecodeError:
                pass
        return s


class WordVocabulary(Vocabulary):
    def __init__(self):
        self.id_to_string = {}
        self.string_to_id = {}
        self.pad_id = self.unk_id = 0 # default to zeroth token by convention

    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            return self.add_new_word(string)
        else:
            return self.pad_id

    def encode(self, text, extend_vocab=False):
        return torch.LongTensor([self.get_idx(char, extend_vocab=extend_vocab) for char in text.split()])

    def decode(self, ids):
        return ' '.join([self.id_to_string[id] for id in ids])

    def format(self, s):
        return s


def tokenize_bytes(text_file, vocab, extend_vocab=False, device='cpu'):
    if vocab is None:
        vocab = Vocabulary.bytes()

    print(f"Memory mapping bytes from: {text_file}", file=sys.stderr)
    s = torch.ByteStorage.from_file(str(text_file), size=Path(text_file).stat().st_size, shared=False)
    data = torch.ByteTensor(s)
    return data, vocab


def load_u16(filename):
    s = torch.ShortStorage.from_file(str(filename), size=Path(filename).stat().st_size // 2, shared=False)
    data = torch.ShortTensor(s)
    print(f"Memory mapping u16 from: {filename}, shape: {data.shape}", file=sys.stderr)
    return data


def tokenize_chars(text_file, vocab, extend_vocab=True, device='cpu'):
    if vocab is None:
        vocab = Vocabulary()

    full_text = []
    print(f"Reading text file from: {text_file}", file=sys.stderr)
    with open(text_file, 'r') as text:
        for line in text:
            tokens = list(line)
            for token in tokens:
                full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

    data = torch.tensor(full_text, device=device, dtype=torch.int16)
    return data, vocab


def tokenize_words(text_file, vocab, extend_vocab=True, device='cpu'):
    if vocab is None:
        vocab = WordVocabulary()

    full_text = []
    print(f"Using word vocabulary from: {text_file}", file=sys.stderr)
    with open(text_file, 'r') as text:
        for line in text:
            for token in line.strip().split():
                full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))
    print(f"Vocabulary size: {len(vocab)} {vocab.string_to_id} {[vocab.get_idx(i, extend_vocab=False) for i in range(32)]}", file=sys.stderr)
    data = torch.tensor(full_text, device=device, dtype=torch.int16)
    return data, vocab


class SymbolTapeNoPad:
    def __init__(self, data, batch_size, bptt_len):
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.tape_len = math.ceil(len(data) / batch_size)
        self.tape_parts, self.trailing_tokens = divmod(self.tape_len, bptt_len)
        self.data = data

    def __len__(self):
        return self.tape_parts + int(bool(self.trailing_tokens))

    def __getitem__(self, i):
        if i == 0:
            # first batch: add pad_id token in the beginning
            batch = self.data.new_full((self.bptt_len, self.batch_size), -2323)

            for tape_index in range(self.batch_size):
                offset = tape_index * (self.tape_len - 1)
                # remove one for padding
                part = self.data[offset + i*self.bptt_len:offset + (i+1) * self.bptt_len]
                batch[:len(part), tape_index] = part
        elif i == self.tape_parts:
            # last batch: truncate
            batch = self.data.new_full((self.trailing_tokens, self.batch_size), -2323)

            for tape_index in range(self.batch_size):
                # remove one for the padding in batch 0
                offset = tape_index * (self.tape_len - 1)
                part = self.data[offset + i*self.bptt_len:offset + i*self.bptt_len + self.trailing_tokens]
                batch[:len(part), tape_index] = part
        else:
            # other batches: account for the padding in batch 0
            batch = self.data.new_full((self.bptt_len, self.batch_size), -2323)

            for tape_index in range(self.batch_size):
                offset = tape_index * (self.tape_len - 1)
                part = self.data[offset + i*self.bptt_len:offset + (i+1) * self.bptt_len]
                batch[:len(part), tape_index] = part

        return batch


class SymbolTape:
    def __init__(self, data, batch_size, bptt_len, pad_id):
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.pad_id = pad_id
        self.tape_len = math.ceil(len(data) / batch_size + 1)
        self.tape_parts, self.trailing_tokens = divmod(self.tape_len, bptt_len)
        self.data = data

    def __len__(self):
        return self.tape_parts + int(bool(self.trailing_tokens))

    def __getitem__(self, i):
        if i == 0:
            # first batch: add pad_id token in the beginning
            batch = self.data.new_full((self.bptt_len, self.batch_size), self.pad_id)

            for tape_index in range(self.batch_size):
                offset = tape_index * (self.tape_len - 1)
                # remove one for padding
                part = self.data[offset + i*self.bptt_len:offset + (i+1) * self.bptt_len - 1]
                batch[1:len(part)+1, tape_index] = part
        elif i == self.tape_parts:
            # last batch: truncate
            batch = self.data.new_full((self.trailing_tokens, self.batch_size), self.pad_id)

            for tape_index in range(self.batch_size):
                # remove one for the padding in batch 0
                offset = tape_index * (self.tape_len - 1) - 1
                part = self.data[offset + i*self.bptt_len:offset + i*self.bptt_len + self.trailing_tokens]
                batch[:len(part), tape_index] = part
        else:
            # other batches: account for the padding in batch 0
            batch = self.data.new_full((self.bptt_len, self.batch_size), self.pad_id)

            for tape_index in range(self.batch_size):
                # remove one
                offset = tape_index * (self.tape_len - 1) - 1
                part = self.data[offset + i*self.bptt_len:offset + (i+1) * self.bptt_len]
                batch[:len(part), tape_index] = part

        return batch


def make_vocab(vocab_descriptor):
    "Possible values: bytes|ascii|cmu|xen|words:path/to/words.txt|path/to/words.txt"

    match vocab_descriptor.split(':', maxsplit=1):
        case ["bytes"]:
            return Vocabulary.bytes()
        case ["ascii"]:
            return Vocabulary.ascii()
        case ["cmu"]:
            return xen.Vocabulary(add_closures=False)
        case ["xen"]:
            return xen.Vocabulary(add_closures=True)
        case ["words", path]:
            _, vocab = tokenize_words(path, None)
            return vocab
        case [path]:
            _, vocab = tokenize_words(path, None)
            return vocab
        case _:
            raise ValueError("Unknown vocabulary descriptor. " + make_vocab.__doc__)


if __name__ == '__main__':
    tape = SymbolTapeNoPad(torch.as_tensor(bytearray(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv")),
                      batch_size=3, bptt_len=8)
                      #batch_size=256, bptt_len=2, pad_id=0)
    for i in range(len(tape)):
        print(tape[i], tape[i].shape)

