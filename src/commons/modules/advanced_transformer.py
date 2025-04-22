import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from commons.utils import linear

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, num_blocks=-1, gain=1.0, dropout=0., inverted=False, epsilon=1e-5):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        if num_blocks != -1:
            assert int(d_model / num_heads) % num_blocks == 0, "d_model must be divisible by num_blocks"
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.inverted = inverted

        self.epsilon = epsilon
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    def forward(self, q, k, v, attn_mask=None, attn_bias=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        if self.num_blocks < 0:
            q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, -2)
            k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, -2)
            v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, -2)
        else:
            q = self.proj_q(q).view(B, T, self.num_heads, self.num_blocks, -1).transpose(1, -2)
            k = self.proj_k(k).view(B, S, self.num_heads, self.num_blocks, -1).transpose(1, -2)
            v = self.proj_v(v).view(B, S, self.num_heads, self.num_blocks, -1).transpose(1, -2)
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn += attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        if self.inverted:
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(B, self.num_heads, T, S)

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float('0.'))

            attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, -2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        return output


class OctavesScalarEncoder(nn.Module):
    def __init__(self, D, max_period):
        super().__init__()
        assert D % 2 == 0
        self.D = D
        self.multipliers = nn.Parameter(
            (2 ** torch.arange(self.D // 2).float()) * 2 * math.pi / max_period,
            requires_grad=False
        )

    def forward(self, scalars):
        """

        :param scalars: *ORI
        :param D:
        :return: *ORI, D
        """
        ORI = scalars.shape
        scalars = scalars.flatten()  # B

        x = scalars[:, None] * self.multipliers[None, :]  # B, D // 2
        x = math.sqrt(2) * torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # B, D
        x = x.reshape(*ORI, self.D)  # *ORI, D

        return x

class TemporalInvertedMultiHeadAttention(nn.Module):

    def __init__(self, d_query, d_value, num_heads, gain=1.0, dropout=0., epsilon=1e-5):
        super().__init__()

        assert d_query % num_heads == 0, "d_query must be divisible by num_heads"
        assert d_value % num_heads == 0, "d_value must be divisible by num_heads"
        self.d_query = d_query        
        self.d_value = d_value
        self.num_heads = num_heads

        self.epsilon = epsilon

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_query, d_query, bias=False)
        self.proj_k = linear(d_value, d_query, bias=False)
        self.proj_v = linear(d_value, d_query, bias=False)
        self.proj_o = linear(d_query, d_query, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None, attn_bias=None):
        """
        q: B, T, N, D
        k: B, T, L, D
        v: B, T, L, D
        attn_mask: TN x TL
        attn_bias: B, H, TN, TL
        return: B, T, N, D
        """
        B, T, N, _ = q.shape
        _, _, L, _ = k.shape

        q = self.proj_q(q).view(B, T * N, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, T * L, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, T * L, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn += attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = attn.reshape(B, self.num_heads, T, N, T * L)
        attn = attn.permute(0, 2, 4, 1, 3).reshape(B, T, T * L, self.num_heads * N)
        attn = F.softmax(attn, dim=-1).reshape(B, T, T * L, self.num_heads, N).permute(0, 3, 1, 4, 2)
        attn = attn.reshape(B, self.num_heads, T * N, T * L)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('0.'))

        attn_vis = attn.detach().clone()
        attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon

        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T * N, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        output = output.reshape(B, T, N, -1)

        return output, attn_vis


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, gain=1.0, dropout=0.0):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads, gain=gain, dropout=dropout)

        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.GELU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout)
        )

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """

        x = self.norm_attn(input)
        input = input + self.attn(x, x, x)

        x = self.norm_ffn(input)
        input = input + self.ffn(x)

        return input


class Transformer(nn.Module):

    def __init__(self, d_model, num_blocks, num_heads, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        gain = (2 * num_blocks) ** (-0.5)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, gain=gain, dropout=dropout)
             for _ in range(num_blocks)])

    def forward(self, input):
        """
        input: batch_size x source_len x d_model
        return: batch_size x source_len x d_model
        """

        input = self.norm(input)
        for block in self.blocks:
            input = block(input)

        return input