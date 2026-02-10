import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional 

from models.rotary_embedding import RotaryEmbedding

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, bias:bool, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(features, bias=bias)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
            # return self.norm(x + self.dropout(sublayer(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float, apply_rotary:bool) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.apply_rotary = apply_rotary

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

        if apply_rotary:
            assert self.d_k % 2 == 0, f"head dimensions must be even"
            self.rotary_emb = RotaryEmbedding(dim = self.d_k)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        if self.apply_rotary:
            query = self.rotary_emb.rotate_queries_or_keys(query)
            key = self.rotary_emb.rotate_queries_or_keys(key)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float, bias:bool) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, bias, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class TransformerEncoderRotary(nn.Module):
    def __init__(self,
                 d_model: int,
                 nlayers: int,
                 nheads: int,
                 dropout: float,
                 d_ff: int,
                 bias: bool,
                 apply_rotary: bool,
                 seqlen: int,
                 project_dims: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, bias=bias)
        encoder_blocks = []
        for _ in range(nlayers):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, nheads, dropout, apply_rotary=apply_rotary)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout, bias)
            encoder_blocks.append(encoder_block)
        self.layers = nn.ModuleList(encoder_blocks)
        self.apply_rotary = apply_rotary
        if not apply_rotary:
            self.pe = PositionalEncoding(d_model, seqlen, dropout)
        
        self.project_dims = project_dims
        if project_dims is not None:
            self.project = nn.Conv1d(in_channels=d_model, out_channels=project_dims, kernel_size=1, stride=1)

    def forward(self, x, mask=None):
        if not self.apply_rotary:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        if self.project_dims is not None:
            x = self.project(x.permute(0,2,1)).permute(0,2,1)
            return F.normalize(x, dim=-1)
        else: 
            return F.normalize(x, dim=-1)

class TransformerEncoderFixed(nn.Module):
    """
    Encoder class. Transforms spectrogram to a sequence of contextual embeddings follwed by their concatenation and projection.
    """
    def __init__(self,
                d_model:int,
                nhead:int,
                dim_feedforward:int,
                num_layers:int,
                seqlen:int,
                project_dims:int,
                activation:Optional[str]="relu",
                dropout:Optional[float]=0,
                bias:Optional[bool]=False):
        super().__init__()
        """
        Parameters:
            d_model: (int), dimension of input embeddings to transformer encoder. Same as dimension of patch embeddings
            patch_size: (tuple, ints), patch size of spectrogram
            nhead: (int), number of attention heads in a self-attention layer
            dim_feedforward: (int), feedforward layer dims. FFN(3-layers): d_model -> dim_feedforward -> d_model
            num_layers: (int), number of self-attention layers in transformer encoder
            activation: (str), activation used in transformer. {"relu", "gelu"}
        """
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=1e-05, batch_first=True, bias=bias)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.sinusoidal_pos_emb = PositionalEncoding(d_model, seqlen, dropout=dropout)
        self.project_dims = project_dims
        if project_dims is not None:
            self.project = nn.Conv1d(in_channels=d_model, out_channels=project_dims, kernel_size=1, stride=1)
        
    def forward(self,x):
        if self.project_dims is not None:
            x = self.encoder(self.sinusoidal_pos_emb(x))
            x = self.project(x.permute(0,2,1)).permute(0,2,1)
            return F.normalize(x, dim=-1)
        else:
           return F.normalize(self.encoder(self.sinusoidal_pos_emb(x)), dim=-1)
