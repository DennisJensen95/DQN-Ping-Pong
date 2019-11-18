import torch
from torch import nn
from torch import optim

class DQN(nn.Module):

    def __init__(self, learning_rate):
        super(DQN, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=80*80, out_features=200, bias=False),
            nn.ReLU()
        )
        out_features = 200

        self.out = nn.Linear(in_features=out_features, out_features=1, bias=False)

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.layer_1(x)
        return nn.Sigmoid()(self.out(x))

    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))

