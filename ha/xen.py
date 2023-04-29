"""
xen is extension of CMUdict that includes TIMIT-style acoustic landmarks compatible with Ukrainian
designed by Mykola Sazhok
"""

import torch

from g2p_en import G2p


class Vocabulary:
    def __init__(self, add_closures=True):
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
                            "V", "W", "Y", "Z", "ZH"]

        if add_closures:
            # add TIMIT-style glottal closures
            self.closures =  {
                'B': ['bcl', 'B'],
                'CH': ['tcl', 'CH'],
                'D': ['dcl', 'D'],
                'G': ['gcl', 'G'],
                'JH': ['dcl', 'JH'],
                'K': ['kcl', 'K'],
                'P': ['pcl', 'P'],
                'T': ['tcl', 'T']
            }
            self.rdictionary.extend(["bcl", "tcl", "dcl", "gcl", "pcl", "kcl"])
        else:
            self.closures = {}

        self.dictionary = {c: i for i, c in enumerate(self.rdictionary, start=1)}

    def state_dict(self):
        return {
            'rdictionary': self.rdictionary,
        }

    def load_state_dict(self, state_dict):
        self.rdictionary = state_dict['rdictionary']
        self.dictionary = {c: i for i, c in enumerate(self.rdictionary, start=1)}

    def __len__(self):
        return len(self.rdictionary) + 1

    def encode(self, text):
        labels = [phoneme.replace('2', '0')
                  for c in self.g2p(text)
                  if c != "'"
                  for phoneme in self.closures.get(c, [c])]
        targets = torch.LongTensor([self.dictionary[phoneme] for phoneme in labels])
        return targets

    def decode(self, labels):
        return ['' if l == 0 else self.rdictionary[l-1] for l in labels]

