import torch
import torch.nn as nn
import torch.nn.functional as F

from g2p_en import G2p


class Encoder(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=1024, vocab_size=55+1):
        super().__init__()
        self.subsample = nn.Conv1d(input_dim, hidden_dim, 3, stride=2, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        x = self.subsample(inputs.mT).mT
        x = x.relu()
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x.relu()
        logits = self.classifier(x)
        return logits    


class Recognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0)
        self.g2p = G2p()

        self.dictionary = [" ",
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
        print('dictionary', len(self.dictionary), "+1 for blank")
        self.dictionary = {c: i for i, c in enumerate(self.dictionary, start=1)}
        
    def encode(self, text):
        labels = self.g2p(text)
        #print(text, labels)
        return torch.LongTensor([self.dictionary[c.replace('2', '0')] for c in labels if c != "'"])

    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        if input_lengths is None:
            input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
        if target_lengths is None:
            target_lengths = torch.full((logits.shape[0],), len(targets), dtype=torch.long)

        logits = logits.permute(1, 0, 2) # T, N, C
        logits = logits.to(torch.float32)
        log_probs = F.log_softmax(logits, dim=-1)
        return self.ctc(log_probs, targets, input_lengths=input_lengths, target_lengths=target_lengths)


if __name__ == '__main__':
    encoder = Encoder()
    reco = Recognizer()
    x = torch.randn(1, 13, 320)
    logits = encoder(x)
    print(logits.shape)
    print(reco(logits, reco.encode("hello world")[None,:]))