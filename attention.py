import torch
from torch import nn
import math

class multiHeadAttention(nn.module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linaer(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, query, key, value, mask=None):
        B, L, D = query.shape
        H = self.num_heads
        d_k = self.embed_dim

        Q = self.q_proj(query).view(B, L, H, d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, H, d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, H, d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out
    
class multiHeadAttention2(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch, seq_len, embed_dim = query.shape
        qkv = self.qkv_proj(query)
        Q, K, V = qkv.chunk(3, dim=-1)

        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) * self.scale
        scores = self.attn_dropout(scores)

        if mask is not None:
            scores.masked_fill(mask==0, 1e-9)
        
        attn = nn.Softmax(scores, dim=-1)

        output = attn @ V
        attn = attn.tranpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        output = self.out_proj(output)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
        
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.multiHeadAttn = multiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ff = FeedForward(embed_dim=embed_dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.multiHeadAttn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = multiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn = multiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ff = FeedForward(embed_dim=embed_dim, hidden_dim=ffn_hidden_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        _x = x
        self_attn = self.self_attn(x, x, x, src_mask)
        x = _x + self.dropout(self_attn)
        x = self.norm1(x)

        _x = x
        cross_attn = self.cross_attn(x, enc_out, enc_out, tgt_mask)
        x = _x + self.dropout(cross_attn) 
        x = self.norm2(x)

        ffn_out = self.ff(x)
        x = x + ffn_out
        x = self.norm3(x)      

        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=512, num_heads=8, ffn_hidden_dim=2048, num_layers=6, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, embed_dim)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dec_layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        for layer in self.enc_layers:
            src = layer(src, src_mask)

        enc_out = src
        for layer in self.dec_layers:
            tgt = layer(tgt, enc_out, src_mask, tgt_mask)
        
        out = self.output_layer(tgt)
        return out






