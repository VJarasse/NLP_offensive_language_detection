import torch.nn as nn


class FFNN(nn.Module):

    def __init__(self):
        super().__init__()
        embedding_dim = 100
        h_dim1 = 512
        h_dim2 = 256
        h_dim3 = 64
        num_classes = 2

        # hidden layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=h_dim1, bias=True),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=h_dim1, out_features=h_dim2, bias=True),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=h_dim2, out_features=h_dim3, bias=True),
            nn.ReLU()
        )

        # output layer
        self.layer4 = nn.Linear(in_features=h_dim3, out_features=num_classes, bias=True)

    def forward(self, x):
        x = x.view(-1, 1, 100)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
