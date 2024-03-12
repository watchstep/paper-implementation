import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    dim_model: int = 512
    dim_ff: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = -1
    dropout: float = 0.1 
    max_batch_size: int = 32
    max_seq_len : int = 2048

class TokenEmbedding(nn.Module):
    def __init__(self, args:ModelArgs):
        self.dim_model = args.dim_model
        self.vocab_size = args.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.dim_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim_model) # scaling
    
class PositionalEncoding(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.encoding = torch.zeros(args.max_seq_len, args.dim_model)
        self.pos = torch.arange(0, args.max_seq_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(args.dim_model, step=2).float()
        self.encoding[:, 0::2] = torch.sin(self.pos / (1e4 ** (_2i / args.dim_model)))
        self.encoding[:, 1::2] = torch.cos(self.pos / (1e4 ** (_2i / args.dim_model)))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        return self.encoding[:seq_len, :]
    
class Attention(nn.Module):
    '''Multi-head self attention (q=k=v)'''
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim_model // args.n_heads # 512 // 8 (64) dim of attention head
        self.w_q = nn.Linear(self.head_dim, self.head_dim) # weight matrix of query
        self.w_k = nn.Linear(self.head_dim, self.head_dim) # weight matrix of key
        self.w_v = nn.Linear(self.head_dim, self.head_dim) # weight matrix of value
        self.w_o = nn.Linear(self.head_dim, self.head_dim) # output (concat each head)
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.w_q(x), self.w_k(x), self.w_v(x) # (batch_size, seq_len, hidden_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # (batch_size, n_heads, seq_len, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_score = (xq @ xk.transpose(2, 3)) / math.sqrt(self.head_dim) 
        if attn_mask is not None:
            attn_score = torch.masked_fill(attn_mask == 0, -1e10)
        attn_score = F.softmax(attn_score, dim=-1) 
        output = torch.matmul(attn_score, xv) # scaled dot product attention
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) # (batch_size, seq_len, head_dim)
        # contiguous() -> view()
        return self.w_o(output)
    
class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim_model, args.dim_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        self.w2 = nn.Linear(args.dim_ff, args.dim_model)
    
    def forward(self, x):
        return self.w2(self.dropout(self.relu(self.w1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.attn = Attention(args) 
        self.ffn = FeedForward(args) 
        self.norm1 = nn.LayerNorm(args.dim_model)
        self.norm2 = nn.LayerNorm(args.dim_model)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.ffn_dropout = nn.Dropout(args.dropout)
    
    def forward(self, x, attn_mask=None):
        attn = self.attn(x, attn_mask) # multi-head attention
        x = self.norm1(attn + x) # add & norm
        x = self.attn_dropout(x)
        ffn = self.ffn(x) # feed forward
        x = self.norm2(ffn + x) # add & norm
        x = self.ffn_dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_layers = args.n_layers 
        self.tok_emb = TokenEmbedding(args)
        self.pos_emb = PositionalEncoding(args)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(self.n_layers)])
    
    def forward(self, x, attn_mask=None):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = F.dropout(tok_emb + pos_emb)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.enc_dec_attn = Attention(args)
        self.ffn = FeedForward(args) 
        self.norm1 = nn.LayerNorm(args.dim_model)
        self.norm2 = nn.LayerNorm(args.dim_model)
        self.norm3 = nn.LayerNorm(args.dim_model)
        self.self_attn_dropout = nn.Dropout(args.dropout)
        self.enc_dec_attn_dropout = nn.Dropout(args.dropout)
        self.ffn_dropout = nn.Dropout(args.dropout)
    
    def forward(self, enc, dec, self_attn_mask=None, enc_dec_attn_mask=None):
        self_attn = self.self_attn(x, self_attn_mask)
        x = self.norm1(self_attn + x)
        x = self.self_attn_dropout(x)
        
        if enc is not None:
            enc_dec_attn = self.enc_dec_attn(x, enc_dec_attn_mask)
            x = self.norm2(enc_dec_attn + x)
            x = self.enc_dec_attn_dropout(x)
        
        ffn = self.ffn(x)
        x = self.norm3(ffn + x)
        x = self.ffn_dropout(x)
        return x
            
class Decoder(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_layers = args.n_layers
        self.tok_emb = TokenEmbedding(args)
        self.pos_emb = PositionalEncoding(args)
        self.layers = nn.ModuleList([DecoderBlock(args) for _ in range(self.n_layers)])
    
    def forward(self, x, attn_mask=None):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        x = F.dropout(tok_emb + pos_emb)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim_model = args.dim_model
        self.head_dim = args.dim // args.n_heads # 512 // 8 (64)
        
        self.tok_emb = TokenEmbedding(args)
        self.pos_emb = PositionalEncoding(args)
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.fc_out = nn.Sequential(
            nn.Linear(args.dim_model, args.dim_ff),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        enc  = self.encoder(x)
        dec = self.decoder(enc=enc)
        return self.fc_out(dec)
       