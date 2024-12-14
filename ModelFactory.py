from abc import ABC, abstractmethod
import torch
import torch.nn as nn

IN_DIM = 'block_input_dim'
OUT_DIM = "block_output_dim"

class CustomModel(ABC, nn.Module):
    """
    Abstract base class for all custom models.
    """
    @abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    def reset_hidden_state(self):
        return None


class CustomTimeSeriesModel(CustomModel):
    def __init__(self, fc_input_size, fc_output_size, time_series_block, **kwargs):
        super(CustomTimeSeriesModel, self).__init__()

        self.hidden_size = kwargs['hidden_size']
        self.num_layers = kwargs['num_layers']
        
        self.fc_in = nn.Linear(fc_input_size, kwargs['input_size'])
        self.time_series_block = time_series_block(**kwargs)
        self.fc_out = nn.Linear(self.hidden_size, fc_output_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_state = None

    def forward(self, x, state):
        x = self.fc_in(x)
        x, new_hidden_state = self.time_series_block(x, state)
        x = self.fc_out(x)
        return x, new_hidden_state
    
    def reset_hidden_states(self):
        self.hidden_state = torch.zeros(
            self.num_layers, 1, self.hidden_size, device=self.device
        )


class CustomRNN(CustomTimeSeriesModel):
    def __init__(self, **kwargs):
        super(CustomRNN, self).__init__(
            fc_input_size=kwargs.pop(IN_DIM),
            fc_output_size=kwargs.pop(OUT_DIM),
            time_series_block=nn.RNN,
            **kwargs
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=1)
        if self.hidden_state is None: super().reset_hidden_states()
        extend_state = self.hidden_state[:, :1, :].repeat(1, x.size(0), 1)
        output, self.hidden_state = super().forward(x, extend_state)
        self.hidden_state = self.hidden_state[:, -1, :]
        return output[:, -1, :]


class CustomGRU(CustomTimeSeriesModel):
    def __init__(self, **kwargs):
        super(CustomGRU, self).__init__(
            fc_input_size=kwargs.pop(IN_DIM),
            fc_output_size=kwargs.pop(OUT_DIM),
            time_series_block=nn.GRU,
            **kwargs
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=1)
        if self.hidden_state is None: super().reset_hidden_states()
        extend_state = self.hidden_state[:, :1, :].repeat(1, x.size(0), 1)
        output, self.hidden_state = super().forward(x, extend_state) 
        self.hidden_state = self.hidden_state[:, -1, :]
        return output[:, -1, :]


class CustomLSTM(CustomTimeSeriesModel):
    def __init__(self, **kwargs):
        super(CustomLSTM, self).__init__(
            fc_input_size=kwargs.pop(IN_DIM),
            fc_output_size=kwargs.pop(OUT_DIM),
            time_series_block=nn.LSTM,
            **kwargs
        )
        self.cell_state = None

    def reset_hidden_states(self):
        super().reset_hidden_states()
        # Initialize cell state for LSTM
        self.cell_state = torch.zeros(
            self.num_layers, 1, self.hidden_size, device=self.device
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=1)
        if self.hidden_state is None or self.cell_state is None: self.reset_hidden_states()
        extend_h_state = self.hidden_state[:, :1, :].repeat(1, x.size(0), 1)
        extend_c_state = self.cell_state[:, :1, :].repeat(1, x.size(0), 1)
        output, (self.hidden_state, self.cell_state) = super().forward(x, (extend_h_state, extend_c_state))
        self.hidden_state = self.hidden_state[:, -1, :]
        self.cell_state = self.cell_state[:, -1, :]
        return output[:, -1, :]


class CustomFullyConnected(CustomModel):
    def __init__(self, block_input_dim, block_output_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(block_input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, block_output_dim))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute position indices and division terms
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # Shape: [d_model/2]

        # Compute sinusoidal positional encodings
        pe = torch.zeros(max_len, d_model)  # Shape: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Register positional encodings as a buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: [1, max_len, d_model]

    def forward(self, x):
        # Add positional encodings to input tensor
        x = x + self.pe[:, :x.size(1), :]
        return x


class CustomTransformer(CustomModel):
    def __init__(
            self, 
            d_model, 
            nhead, 
            num_encoder_layers, 
            num_decoder_layers, 
            dim_feedforward, 
            block_input_dim,
            block_output_dim,
            max_length=5000,
        ):
        super().__init__()
        self.positional_encoder = PositionalEncoding(d_model, max_length)
        self.fc_in = nn.Linear(block_input_dim, d_model)
        self.fc_out = nn.Linear(d_model, block_output_dim)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

    def forward(self, src, tgt):
        src = self.fc_in(src)
        tgt = self.fc_in(tgt)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        output = self.transformer(src, tgt)

        output = self.fc_out(output)
        return output   

class CustomMultiheadAttention(CustomModel):
    def __init__(
            self, 
            embed_dim, 
            num_heads, 
            block_input_dim,
            block_output_dim,
            max_length=5000
        ):
        super().__init__()
        self.fc_in = nn.Linear(block_input_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, block_output_dim)
        self.positional_encoder = PositionalEncoding(embed_dim, max_length)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        query = self.positional_encoder(self.fc_in(query))
        key = self.positional_encoder(self.fc_in(key))
        value = self.positional_encoder(self.fc_in(value))

        attn_output, attn_weights = self.attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        output = self.fc_out(attn_output)
        return output, attn_weights
    
class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "fully_connected":
            return CustomFullyConnected(**kwargs)

        kwargs['batch_first'] = True
        if model_type == "rnn":
            return CustomRNN(**kwargs)
        if model_type == "gru":
            return CustomGRU(**kwargs)
        if model_type == "lstm":
            return CustomLSTM(**kwargs)
        if model_type == "transformer":
            return CustomTransformer(**kwargs)
        if model_type == "multihead_attention":
            return CustomMultiheadAttention(**kwargs)

        raise ValueError(f"Unknown model type: {model_type}")
