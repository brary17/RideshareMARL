import torch
from network import Network

class Critic(Network):
    def __init__(
            self, 
            state_dim, 
            model_type,            
        ):
        super(Critic, self).__init__(
            input_dim=state_dim, 
            output_dim=1, 
            model_type=model_type,            
        )

    def save(self, path):
        super().save("cricit_" + path)
    
    def load(self, path):
        super().load("cricit_" + path)