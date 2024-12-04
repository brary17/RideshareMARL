import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
            self, 
            layer_num, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            activation_function=torch.relu, 
            last_activation=None, 
            use_rnn=False, 
            rnn_type='LSTM'
        ):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        self.use_rnn = use_rnn

        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)
        self.layers = nn.ModuleList([nn.Linear(layers_unit[i], layers_unit[i + 1]) for i in range(len(layers_unit) - 1)])

        if use_rnn:
            rnn_module = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
            self.rnn = rnn_module(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        else:
            self.rnn = None

        self.last_layer = nn.Linear(layers_unit[-1], output_dim)
        self.network_init()

    def forward(self, x, hidden_state=None):
        if len(x.shape) == 2:  # Single state: [batch_size, state_dim]
            x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, state_dim]

        if self.use_rnn:
            x, hidden_state = self.rnn(x, hidden_state)  # RNN expects [batch, seq, feature]
            x = x[:, -1, :]  # Use the output of the last sequence element

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.last_layer(x)
        if self.last_activation is not None:
            x = self.last_activation(x)

        return x, hidden_state if self.use_rnn else x


    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()
