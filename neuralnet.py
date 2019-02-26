import torch.nn as nn
import torch.nn.functional as F


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
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=h_dim3, out_features=num_classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 100)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class CNN(nn.Module):

    def __init__(self, embedding_dim, out_channels, window_size, output_dim, dropout):
        super(CNN, self).__init__()

        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)

        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(window_size, embedding_dim))

        # the dropout layer
        self.dropout = nn.Dropout(dropout)

        # the output layer
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        # (batch size, max sent length, embedding dim)

        # We unsqueeze one dimension to give space to the coming convolution channels
        embedded = x.unsqueeze(1)

        # (batch size, 1, max sent length, embedding dim)

        feature_maps = self.conv(embedded)

        # (batch size, n filters, max input length - window size +1)

        feature_maps = feature_maps.squeeze(3)

        feature_maps = F.relu(feature_maps)

        # the max pooling layer
        pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])

        pooled = pooled.squeeze(2)

        # (batch size, n_filters)

        dropped = self.dropout(pooled)

        preds = self.fc(dropped)

        preds = torch.sigmoid(preds)

        return preds