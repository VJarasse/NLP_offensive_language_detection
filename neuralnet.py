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


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout, seq_len):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = 32

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            dropout=dropout, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(seq_len * hidden_dim * 2, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).to(device)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).to(device)))

    def forward(self, x):
        if x.shape[0] != self.batch_size:
            pad = torch.zeros((self.batch_size - x.shape[0], x.shape[1], x.shape[2]))
            x = torch.cat((x, pad), dim=0)
        # Shape of x  torch.Size([32, 105, 100])
        # Shape of LSTM out  torch.Size([32, 105, 40])

        # lstm out should be (seq_len, batch, num_directions * hidden_size)
        # elements of self.hidden should be (num_layers * num_directions, batch, hidden_size)

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out.view(-1, self.seq_len * 2 * self.hidden_dim)

        tag_space = self.linear(lstm_out)
        tag_scores = torch.sigmoid(tag_space)
        return tag_scores


class BiLSTMConv(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout, seq_len,
                 channels, window_size):
        super(BiLSTMConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.batch_size = 32

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            dropout=dropout, batch_first=True, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden = self.init_hidden()

        self.conv = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(window_size, 2 * hidden_dim))

        # the dropout layer
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(channels, output_dim)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).to(device)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).to(device)))

    def forward(self, x):
        if x.shape[0] != self.batch_size:
            pad = torch.zeros((self.batch_size - x.shape[0], x.shape[1], x.shape[2]))
            x = torch.cat((x, pad), dim=0)
        # Shape of x  torch.Size([32, 105, 100])
        # Shape of LSTM out  torch.Size([32, 105, 40])

        # lstm out should be (seq_len, batch, num_directions * hidden_size)
        # elements of self.hidden should be (num_layers * num_directions, batch, hidden_size)

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # lstm_out = lstm_out.contiguous()
        # lstm_out = lstm_out.view(-1, self.seq_len * 2 * self.hidden_dim)

        # make space for convolution channels
        lstm_out = lstm_out.unsqueeze(1)
        lstm_out = F.relu(lstm_out)

        conv_out = self.conv(lstm_out)

        conv_out = conv_out.squeeze(3)

        pooled = F.max_pool1d(conv_out, conv_out.shape[2])

        pooled = pooled.squeeze(2)

        # (batch size, n_filters)
        dropped = self.dropout(pooled)

        preds = self.linear(dropped)
        preds = torch.sigmoid(preds)

        return preds
