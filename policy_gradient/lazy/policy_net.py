import torch.nn as nn
import torch.nn.functional as F


class policyNet(nn.Module):
    def __init__(self):
        super.__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, state):
        state = self.affine1(state)
        state = self.dropout(state)
        state = nn.ReLU(state)
        action_scores = self.affine2(state)
        return F.softmax(action_scores, dim=1)
