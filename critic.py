import torch
import torch.nn as nn
from network import Network

class Critic(Network):
    def __init__(
            self, 
            layer_num, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            activation_function, 
            last_activation=None
        ):
        super(Critic, self).__init__(
            layer_num, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            activation_function, 
            last_activation
        )

    def forward(self, *x, hidden_state=None):
        x = torch.cat(x, -1)
        return super().forward(x, hidden_state)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

