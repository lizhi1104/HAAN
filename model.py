import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dropout_rate):
        super(Model, self).__init__()
        self.hidden_layer = nn.Linear(n_in, n_hidden)
        self.out_layer = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.hidden_layer(inputs))
        if is_training:
            x = self.dropout(x)
        return x, self.out_layer(x)
