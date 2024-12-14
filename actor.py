import torch
from network import Network


class Actor(Network):
    def __init__(
            self,
            state_dim, 
            action_dim, 
            model_type,            
            trainable_std,
            **kwargs
        ):
        super(Actor, self).__init__(
            input_dim=state_dim, 
            output_dim=action_dim, 
            model_type=model_type,            
        )
        self.trainable_std = trainable_std
        if self.trainable_std:
            self.logstd = torch.nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        mu = super().forward(x)
        if self.trainable_std:
            std = torch.exp(self.logstd)
            std = std.repeat(mu.size(0), 1)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu, std

    def save(self, path):
        super().save("actor_" + path)
    
    def load(self, path):
        super().load("actor_" + path)