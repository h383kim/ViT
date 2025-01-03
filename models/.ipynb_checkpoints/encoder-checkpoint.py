import torch
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        # Decoder Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(p=p_dropout)

        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

    
    def forwrad(self, x):
        """
        Args:
            x (torch.Tensor): Target sequence tensor of shape (batch_size, patch_len, d_model).
        Returns:
            torch.Tensor: Transformed output tensor of shape (batch_size, patch_len, d_model).
            torch.Tensor: Self-attention scores for the sequence.
        """
        # Self-Attention with residual connection and normalization (Note: Normalize First!)
        normed_x = self.norm1(x)
        attn_context, attn_score = self.self_attention(normed_x, normed_x, normed_x)
        attn_context = self.dropout1(attn_context)
        x = x + attn_context

        # Feed-forward network with residual connection and normalization (Note: Normalize First!)
        normed_x = self.norm2(x)
        residual = self.ffn(normed_x)
        residual = self.dropout2(residual)
        x = x + residual

        return x, attn_score


class Encoder(nn.Module):
    def __init__(self, patch_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        # Create positional embedding
        self.pos_embedding = nn.Parameter(0.02*torch.randn(patch_len, d_model))
        self.dropout = nn.Dropout(p=p_dropout)

        # Create encoder blocks
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, d_ffn, num_heads, p_dropout) for _ in range(num_blocks)])

        # LayerNorm for the CLS patch
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, save_attn_pattern=False):
        """
        Args:
            x (torch.Tensor): Sequence tensor of shape (batch_size, patch_len, d_model).
            save_attn_pattern (bool): If True, saves and returns attention patterns for visualization.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, d_model).
            torch.Tensor: (Optional) Self-attention patterns from all encoder blocks.
        """
        x = x + self.pos_embedding.expand_as(x)
        x = self.dropout(x)

        attn_patterns = torch.tensor([]).to(DEVICE)
        for block in self.enc_blocks:
            x, attn_score = block(x)
            # (Optional) if save_attn_pattern is True, save these and return for visualization/investigation
            if save_attn_pattern:
                attn_patterns = torch.cat([attn_patterns, attn_pattern[0].unsqueeze(0)], dim=0)

        x = x[:, 0, :] # Slicing [CLS] Patch
        x = self.norm(x)

        return x, attn_patterns
        