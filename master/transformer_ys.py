import math
import torch
import torch.nn as nn
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self,emb_size, dropout, maxlen = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size) ##
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) ## [maxlen, 1]
        pos_embedding = torch.zeros((maxlen, emb_size)) ##[maxlen, emb_size]
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) ## (maxlen, 1, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) ## positional encoding은 학습에서 제외

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :]) ##[max_len, batch, hidden]


encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8) ## encoder layer 설정
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12) ## transformer encoder layer만