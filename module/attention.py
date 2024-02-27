#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """
        print("********* multi head attention")
        w, bsz, embed_dim = query.size() #torch.Size([314, 140, 128]) or ([314, 70, 128])?????????
        print('query', query.shape )
        head_dim = embed_dim // self.num_heads # 16 = 128 / 8
        print('head dim self.num_heads',head_dim, self.num_heads)
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        if torch.equal(query, key) and torch.equal(key, value): # self?
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)# (314, 140, 128)
            print('query.shape, q.shpae, self.in_proj_weight.shape, self.in_proj_bias.shape', query.shape, q.shape, self.in_proj_weight.shape, self.in_proj_bias.shape)#torch.Size([314, 140, 128]) torch.Size([314, 140, 128]) torch.Size([384, 128]) torch.Size([384])
        elif torch.equal(key, value):# cross?
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        # project to find q_r, k_r
        if pos_enc is not None:
            # reshape pos_enc
            print('pos_enc, pos_indexes', pos_enc.shape, pos_indexes.shape) # torch.Size([627, 128]) torch.Size([98596])
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w, -1)  # 2W-1xC -> WW'xC -> WxW'xC torch.Size([314, 314, 128]) 
            print('pos_enc', pos_enc.shape, )
            # compute k_r, q_r
            _start = 0
            _end = 2 * embed_dim  # 256
            _w = self.in_proj_weight[_start:_end, :] #torch.Size([256, 128])
            _b = self.in_proj_bias[_start:_end] #torch.Size([256])
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC torch.Size([314, 314, 128]) torch.Size([314, 314, 128])
            print('_end, -w, _b, q_r, k_r', _end, _w.shape, _b.shape, q_r.shape, k_r.shape)
               
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5  # 1/4 아래 einsum을 예상해 미리 4로 나눔
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)  # WxNxExC torch.Size([314, 140, 8, 16])
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)  # WxW'xExC  torch.Size([314, 314, 8, 16]) 
        if k_r is not None:
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'  torch.Size([140, 8, 314, 314])
        print('attn_feat', attn_feat.shape)

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW' torch.Size([140, 8, 314, 314])
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'
            print('attn_feat_pos', attn_feat_pos.shape)

            # 0.1 s
            attn = attn_feat + attn_feat_pos + attn_pos_feat #torch.Size([140, 8, 314, 314])
            print('attn', attn.shape)
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask

        if attn_mask is not None: # None
            attn_mask = attn_mask[None, None, ...]
            attn += attn_mask

        # raw attn
        raw_attn = attn ##torch.Size([140, 8, 314, 314])
        print('attn', attn.shape)

        # softmax
        attn = F.softmax(attn, dim=-1) ##torch.Size([140, 8, 314, 314])
        print('attn', attn.shape)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))  # NxExWxW', W'xNxExC -> NExWxC torch.Size([1120, 314, 16])
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        print('v_o', v_o.shape)
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim) #torch.Size([314, 140, 128])
        print('v_o', v_o.shape)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias) #torch.Size([314, 140, 128])
        print('v_o', v_o.shape)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads #torch.Size([140, 314, 314])
        print('attn', attn.shape)

        # raw attn
        raw_attn = raw_attn.sum(dim=1) #torch.Size([140, 314, 314])
        print('raw_attn', raw_attn.shape)

        return v_o, attn, raw_attn
