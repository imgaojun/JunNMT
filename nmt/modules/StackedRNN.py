import torch
import torch.nn as nn

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)

        self.gamma_ih = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.eps = 0

    def _layer_norm_x(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        std = x.std(1).expand_as(x)
        return g.expand_as(x) * ((x - mean) / (std + self.eps)) + b.expand_as(x)

    def _layer_norm_h(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        return g.expand_as(x) * (x - mean) + b.expand_as(x)

    def forward(self, x, h):

        ih_rz = self._layer_norm_x(
            torch.mm(x, self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
            self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
            self.bias_ih.narrow(0, 0, 2 * self.hidden_size))

        hh_rz = self._layer_norm_h(
            torch.mm(h, self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
            self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
            self.bias_hh.narrow(0, 0, 2 * self.hidden_size))

        rz = torch.sigmoid(ih_rz + hh_rz)
        r = rz.narrow(1, 0, self.hidden_size)
        z = rz.narrow(1, self.hidden_size, self.hidden_size)

        ih_n = self._layer_norm_x(
            torch.mm(x, self.weight_ih.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
            self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
            self.bias_ih.narrow(0, 2 * self.hidden_size, self.hidden_size))

        hh_n = self._layer_norm_h(
            torch.mm(h, self.weight_hh.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
            self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
            self.bias_hh.narrow(0, 2 * self.hidden_size, self.hidden_size))

        n = torch.tanh(ih_n + r * hh_n)
        h = (1 - z) * n + z * h
        return h

class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout, layer_norm=False):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if layer_norm:
                self.layers.append(LayerNormGRUCell(input_size, rnn_size))
                
            else:
                self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)