import torch
from torch import nn

class FFN(nn.Module):
    """
    A Position-wise feed-forward transformation to embeddings.
    Args:
        d_model (int): Dimensionality of the input and output features. 
        d_hidden (int): Dimensionality of the hidden layer.  (4 * d_model in general)
        p_dropout (float): Dropout probability for regularization.
    """
    def __init__(self, d_model, d_hidden, p_dropout):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(d_hidden, d_model)
        )
            
    def forward(self, x):
        '''
        both input x and output tensor has shape (batch_size, seq_len, d_model)
        '''
        return self.ffn(x)