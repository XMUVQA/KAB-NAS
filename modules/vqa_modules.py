from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


class MHAtt(nn.Module):
    def __init__(self, in_channels, out_channels, split_num,dropout_rate=0):
        super(MHAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_num = split_num

        self.linear_v = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_merge = nn.Linear(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.split_num,
            int(self.out_channels/self.split_num)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.split_num,
            int(self.out_channels / self.split_num)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.split_num,
            int(self.out_channels / self.split_num)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.out_channels
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)



class MHSEAtt(nn.Module):
    def __init__(self, in_channels,out_channels,split_num,dropout_rate=0.2):
        super(MHSEAtt, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_num = split_num

        self.linear_v = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_merge = nn.Linear(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        

    def forward(self, v, k, q, mask,semask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(n_batches,
                -1,
                self.split_num,
                int(self.out_channels/self.split_num)
            ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.split_num,
            int(self.out_channels / self.split_num)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.split_num,
            int(self.out_channels / self.split_num)
        ).transpose(1, 2)

        atted = self.seatt(v, k, q, mask,semask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.out_channels
        )

        atted = self.linear_merge(atted)

        return atted

       

    def seatt(self, value, key, query, mask,mask1):

        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        scores = scores.masked_fill(mask,-1e9)
        scores1 = scores.masked_fill(mask1,-1e9)
        att_map1 = F.softmax(scores1, dim=-1)
        att_map1 = self.dropout1(att_map1)

        att1 = torch.matmul(att_map1, value)

       
        return att1 


class FFN(nn.Module):
    def __init__(self, in_channels,mid_channels,out_channels,dropout_rate):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=in_channels,
            mid_size=mid_channels,
            out_size=out_channels,
            dropout_r=dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)