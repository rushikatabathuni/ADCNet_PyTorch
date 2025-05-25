import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, adjoin_matrix=None):
    """
    q, k, v: shape (batch_size, num_heads, seq_len, depth)
    mask: shape broadcastable to (batch_size, num_heads, seq_len, seq_len)
    adjoin_matrix: shape (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
    """
    # Compute scaled dot product attention scores
    # q @ k^T
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # shape: (batch_size, num_heads, seq_len, seq_len)
    
    dk = q.size(-1)
    scale_factor = torch.sqrt(torch.tensor(dk, dtype=q.dtype, device=q.device))
    scaled_attention_logits = matmul_qk / scale_factor

    # Apply mask if given
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Prepare adjacency matrix to be broadcastable over heads
    if adjoin_matrix is not None:
        if adjoin_matrix.dim() == 3:
            # shape (batch_size, seq_len, seq_len) -> add heads dim
            adjoin_matrix = adjoin_matrix.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        scaled_attention_logits += adjoin_matrix

    # Softmax to get attention weights
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    # Attention output
    output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)

    return output, attention_weights

class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,num_heads):
    super().__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    self.depth = d_model // self.num_heads

    self.wq = nn.Linear(d_model,d_model)
    self.wk = nn.Linear(d_model,d_model)
    self.wv = nn.Linear(d_model,d_model)
    self.dense = nn.Linear(d_model,d_model)

  def forward(self,v,k,q,mask,adjoin_matrix):
    batch_size = q.shape[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q,batch_size)
    k = self.split_heads(k,batch_size)
    v = self.split_heads(v,batch_size)

    scaled_attention,attention_weights = scaled_dot_product_attention(q,k,v,mask,adjoin_matrix)
    scaled_attention = scaled_attention.permute(0,2,1,3)
    concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
    output = self.dense(concat_attention)
    return output,attention_weights

  def split_heads(self,x,batch_size):
    x = torch.reshape(x,(batch_size,-1,self.num_heads,self.depth))
    x = x.permute(0,2,1,3)
    return x


class FeedForwardNetwork(nn.Module):
  def __init__(self,dff,d_model):
    super().__init__()
    self.fc1 = nn.Linear(d_model,dff)
    self.fc2 = nn.Linear(dff,d_model)

  def forward(self,x):
    x = F.gelu(self.fc1(x))
    x = self.fc2(x)
    return x


class EncoderLayer(nn.Module):
  def __init__(self,d_model,num_heads,dff,rate = 0.1):
    super().__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = FeedForwardNetwork(dff,d_model)
    self.layernorm1 = torch.nn.LayerNorm(d_model)
    self.layernorm2 = torch.nn.LayerNorm(d_model)

    self.dropout1 = torch.nn.Dropout(rate)
    self.dropout2 = torch.nn.Dropout(rate)

  def forward(self,x,training,mask, adjoin_matrix):
    attn_output , attention_weights = self.mha(x,x,x,mask,adjoin_matrix)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)
    return out2,attention_weights


class Encoder(nn.Module):
  def __init__(self,num_layers, d_model, num_heads,dff,input_vocab_size,maximum_position_encoding, rate = 0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = nn.Embedding(input_vocab_size,d_model)
    self.enc_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)])
    self.dropout = nn.Dropout(rate)

  def forward(self,x,training,mask,adjoin_matrix):
    seq_len = x.shape[1]
    adjoin_matrix = adjoin_matrix.unsqueeze(1)
    x = self.embedding(x)
    x *= torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype, device=x.device))
    x = self.dropout(x)
    for layer in self.enc_layers:
      x, attention_weights = layer(x,training,mask,adjoin_matrix)
    return x,attention_weights

class PredictModel(nn.Module):
  def __init__(self,num_layers = 6, d_model = 256, dff = 512, num_heads = 8, vocab_size = 18, dropout_rate = 0.1):
    super().__init__()
    self.encoder = Encoder(num_layers = num_layers, d_model = d_model,num_heads = num_heads, dff = dff, input_vocab_size = vocab_size, maximum_position_encoding = 200, rate = dropout_rate)
    self.fc1 = nn.Linear(4353,d_model)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.fc2 = nn.Linear(d_model,1)

  def forward(self,x1, adjoin_matrix1,mask1, x2, adjoin_matrix2, mask2,t1,t2,t3,t4, training = False):
    x1,attention_weights1 = self.encoder(x1, training = training, mask = mask1, adjoin_matrix = adjoin_matrix1)
    x1 = x1[:,0,:]
    x2,attention_weights2 = self.encoder(x2, training = False, mask = mask2, adjoin_matrix = adjoin_matrix2)
    x2 = x2[:,0,:]
    x = torch.cat((x1,x2),dim = 1)
    x = torch.cat((x,t1),dim = 1)
    x = torch.cat((x,t2),dim = 1)
    x = torch.cat((x,t3),dim= 1)
    x = torch.cat((x,t4),dim = 1)
    x = self.fc1(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    return x