__author__ = "Jie Lei"

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hsz, n_layers, elu=False):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        layers = []
        prev_dim = in_dim
        if elu:
            non_linearity = nn.ELU(True)
        else:
            non_linearity = nn.ReLU(True)

        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    non_linearity,
                    # nn.Dropout(0.5)
                ])
                prev_dim = hsz

        self.main = nn.Sequential(*layers)
        # self.reset()

    def reset(self):
        for i in range(self.n_layers):
            if isinstance(self.main[i] , nn.Linear):
                kaiming_uniform_(self.main[i].weight)


    def forward(self, x):
        return self.main(x)


if __name__ == '__main__':
    test_in = torch.randn(10, 300)

    mlp1 = MLP(300, 1, 100, 1)
    print("="*20)
    print(mlp1)
    print(mlp1(test_in).size())

    mlp2 = MLP(300, 10, 100, 2)
    print("="*20)
    print(mlp2)
    print(mlp2(test_in).size())

    mlp3 = MLP(300, 5, 100, 4)
    print("=" * 20)
    print(mlp3)
    print(mlp3(test_in).size())
