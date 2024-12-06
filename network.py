import torch
import torch.nn as nn
from ModelFactory import ModelFactory
import json


class Network(nn.Module):
    def __init__(
            self, 
            input_dim,
            output_dim,
            model_type
        ):
        super(Network, self).__init__()
        kwargs = json.load(open('default_model_config.json'))[model_type]
        kwargs['block_input_dim'] = input_dim
        kwargs['block_output_dim'] = output_dim
        self.model = ModelFactory.create_model(**kwargs)
        self.network_init()

    def forward(self, x, *args, hidden_state=None, **kwargs):
        if hidden_state is None: 
            return self.model(x), None

        return self.model(x, hidden_state)
    
    def network_init(self):
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
