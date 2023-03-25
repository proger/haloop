"""
xen is an extension of CMUdict that includes TIMIT-style acoustic landmarks
designed by Mykola Sazhok
"""

import torch

from g2p_en import G2p


def cmu_to_xen(phoneme):
    return {
        'B': ['bcl', 'B'],
        'CH': ['tcl', 'CH'],
        'D': ['dcl', 'D'],
        'G': ['gcl', 'G'],
        'JH': ['dcl', 'JH'],
        'K': ['kcl', 'K'],
        'P': ['pcl', 'P'],
        'T': ['tcl', 'T']
    }.get(phoneme, [phoneme])


class Vocabulary:
    def __init__(self):
        self.g2p = G2p()

        # http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        self.rdictionary = [" ",
                            "AA0", "AA1", "AE0", "AE1", "AH0", "AH1", "AO0", "AO1", "AW0", "AW1", "AY0", "AY1",
                            "B", "CH", "D", "DH",
                            "EH0", "EH1", "ER0", "ER1", "EY0", "EY1",
                            "F", "G", "HH",
                            "IH0", "IH1", "IY0", "IY1",
                            "JH", "K", "L", "M", "N", "NG",
                            "OW0", "OW1", "OY0", "OY1",
                            "P", "R", "S", "SH", "T", "TH",
                            "UH0", "UH1", "UW0", "UW1",
                            "V", "W", "Y", "Z", "ZH",
                            "bcl", "tcl", "dcl", "gcl", "pcl", "kcl"]
        self.dictionary = {c: i for i, c in enumerate(self.rdictionary, start=1)}

    def __len__(self):
        return len(self.rdictionary) + 1

    def encode(self, text):
        labels = self.g2p(text)
        return torch.LongTensor([
            phoneme
            for c in labels
            if c != "'"
            for phoneme in cmu_to_xen(self.dictionary[c.replace('2', '0')])])

    def decode(self, labels):
        return ['' if l == 0 else self.rdictionary[l-1] for l in labels]

