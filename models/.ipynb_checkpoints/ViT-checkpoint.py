import torch
from torch import nn
from einops import rearrange

"""
A PyTorch implementation of the Vision Transformer (ViT).

Args:
    img_size (int): The size of the input image (assumed to be square).
    patch_size (int): The size of the patches to divide the input image into.
    num_blocks (int): The number of encoder blocks in the transformer.
    d_model (int): The dimensionality of the embeddings and transformer.
    d_ffn (int): The dimensionality of the feed-forward network within the encoder.
    num_heads (int): The number of attention heads in the multi-head attention mechanism.
    p_dropout (float): The dropout probability used in the transformer layers.
    num_classes (int): The number of classes for the output classification head. Default is 1000.
    d_ffn_finetune (int or None): If provided, specifies the intermediate layer dimension
                                  for fine-tuning the classification head. If `None`, 
                                  a single linear layer is used in the head.
"""
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_blocks, d_model, d_ffn, num_heads, p_dropout, num_classes=1000, d_ffn_finetune=None):
        super().__init__()

        # Dimensionality of the model's patch embeddings
        self.d_model = d_model
        # Learnable [CLS] token used for classification
        self.CLS_Token = nn.Parameter(torch.zeros(d_model))

        # Calculate the number of patches along with the [CLS] token
        patch_len = ((img_size // patch_size) ** 2) + 1

        # Patch embedding layer (convolution-based)
        self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        # Transformer encoder
        self.encoder = Encoder(patch_len, num_blocks, d_model, d_ffn, num_heads, p_dropout)

        # Classification head
        if d_ffn_finetune is not None:
            self.head = nn.Linear(d_model, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, d_ffn_finetune),
                nn.Tanh(),
                nn.Linear(d_ffn_finetune, num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where:
                              N = batch size,
                              C = number of channels (3 for RGB images),
                              H = height of the input image,
                              W = width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (N, num_classes), where:
                          N = batch size,
                          num_classes = number of output classes.
        """
        # Apply patch embedding (convert image into patches)
        x = self.patch_embedding(x) # # x: (N, d_model, H/patch_size, W/patch_size)
        # Rearrange patches to create patch sequences
        x = rearrange(x, 'N C p_H p_W -> N (p_H p_W) C') # x: (N, patch_len, d_model)

        # Expand [CLS] token to match batch size and concatenate with patch embeddings
        CLS_Tokens = self.CLS_Token.expand(x.shape[0], 1, -1) # CLS_Tokens: (N, 1, d_model)
        x = torch.cat([CLS_Tokens, x], dim=1) # x: (N, patch_len + 1, d_model)

        # Pass through the transformer encoder
        enc_out, attn_patterns = self.encoder(x)

        # Pass the encoder's output through the classification head
        x = self.head(enc_out)

        return x