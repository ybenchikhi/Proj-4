import torch
import math
import torch.nn.functional as F
#import torch.Tensor as T



class CustomSelfAttention(torch.nn.Module):
    '''
    Custom self-attention implementation.

    Args:
        w_q: Query weight matrix of shape (hidden_dim, hidden_dim)
        w_k: Key weight matrix of shape (hidden_dim, hidden_dim)
        w_v: Value weight matrix of shape (hidden_dim, hidden_dim)
        w_o: Output weight matrix of shape (hidden_dim, hidden_dim)
        num_heads: Number of attention heads
    '''
    def __init__(self, w_q, w_k, w_v, w_o, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o

    def forward(self, x, causal=False):
        '''
        Forward pass for the self-attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        '''
    
        batch_size, seq_len, _ = x.shape

        # Obtain the query, key, and value tensors
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        ##############################################################

        # TODO: Implement the self-attention operation

        q_out = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k_out = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v_out = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        #when u call transpose u need to specify the actual like indices to be swapped
        '''q_out.transpose(self.seq_len, self.num_heads)
        k_out.transpose(self.seq_len, self.num_heads)
        v_out.transpose(self.seq_len, self.num_heads)

        #softmax should operate across the key dimesnion (the last dimensiom)

        #most tensor operations don't change the tensor in place u need to assign an output
        
        '''
        #t_out = t_out.transpose(-2, -1)
        score_matrix = (q_out @ k_out.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if causal == True:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype = torch.bool, device = x.device), diagonal = 1,)
            score_matrix = score_matrix.masked_fill(mask, float('-inf'));

        attn_weights = F.softmax(score_matrix, dim = -1)
        o = attn_weights @ v_out

        #raise NotImplementedError

        ##############################################################

        # Concatenate attention heads
        o = o.transpose(1, 2)
        o = o.contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Apply output projection
        o = o @ self.w_o.T
        return o