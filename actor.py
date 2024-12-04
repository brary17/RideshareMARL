import torch
import torch.nn as nn
from network import Network


class Actor(Network):
    def __init__(
            self,
            layer_num,
            input_dim,
            output_dim,
            hidden_dim,
            activation_function=torch.tanh,
            last_activation=None,
            trainable_std=False,
            use_rnn=False,
            rnn_type='LSTM',
        ):
        super(Actor, self).__init__(
            layer_num, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            activation_function, 
            last_activation,
            use_rnn, 
            rnn_type
        )
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x, hidden_state=None):
        mu, hidden_state = super().forward(x, hidden_state)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu, std, hidden_state

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
