from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def reset_hidden_states(self):
        pass
