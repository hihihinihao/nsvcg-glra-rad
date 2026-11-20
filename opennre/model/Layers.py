"""Transformer layers for the reconstructed Relation-Aware Denoiser.

The released NSVCG tree imports eight names from ``opennre.model.Layers`` but does
not ship the file.  The paper (Sec. 3.5) describes a conditional denoiser: a noisy
relation distribution is embedded, refined by N attention blocks guided by the
multimodal condition, and decoded by an MLP.  The calling convention recovered
from ``softmax_nn.py`` is::

    layer = DecoderLayerPre_woSA(d_model, d_inner, n_head, d_k, d_v)
    out, attn, _ = layer(x, memory)      # x: (B, 1, d_model), memory: (B, L, d_model)

so the only layer actually instantiated by the release is a pre-LayerNorm decoder
block that cross-attends from the single noisy token to the condition and then
applies a position-wise feed-forward network.  The remaining names are provided
for import compatibility.
"""
import torch
import torch.nn as nn

from .attention import MultiheadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, return_attns=False):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        if return_attns:
            return enc_output, enc_slf_attn
        return enc_output, enc_slf_attn, None


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None,
                dec_enc_attn_mask=None, return_attns=False):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        if return_attns:
            return dec_output, dec_slf_attn, dec_enc_attn
        return dec_output, dec_enc_attn, None


class DecoderLayer_woSA(nn.Module):
    """Decoder block without self-attention (post-LayerNorm)."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.enc_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_enc_attn_mask=None, return_attns=False):
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn, None


class DecoderLayerPre_woSA(nn.Module):
    """Pre-LayerNorm decoder block without self-attention.

    This is the block instantiated by ``DenoiseNetwork`` (module_id == 0): the
    noisy distribution token cross-attends to the multimodal condition.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.enc_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, dec_enc_attn_mask=None, return_attns=False):
        residual = dec_input
        x = self.enc_attn_norm(dec_input)
        x, dec_enc_attn = self.enc_attn(x, enc_output, enc_output, mask=dec_enc_attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.ffn_norm(x)
        x = self.pos_ffn(x)
        x = residual + self.dropout(x)
        return x, dec_enc_attn, None


class DecoderLayerPre_Enc(nn.Module):
    """Pre-LayerNorm decoder block with self-attention on the noisy token."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.slf_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None,
                dec_enc_attn_mask=None, return_attns=False):
        residual = dec_input
        x = self.slf_attn_norm(dec_input)
        x, _ = self.slf_attn(x, x, x, mask=slf_attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.enc_attn_norm(x)
        x, attn = self.enc_attn(x, enc_output, enc_output, mask=dec_enc_attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.ffn_norm(x)
        x = self.pos_ffn(x)
        x = residual + self.dropout(x)
        return x, attn, None


class DecoderLayer_Enc(nn.Module):
    """Post-LayerNorm variant of :class:`DecoderLayerPre_Enc`."""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None,
                dec_enc_attn_mask=None, return_attns=False):
        x, _ = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        x, attn = self.enc_attn(x, enc_output, enc_output, mask=dec_enc_attn_mask)
        x = self.pos_ffn(x)
        return x, attn, None


class DecoderLayer_context(DecoderLayer):
    """Alias kept for import compatibility with the released tree."""
