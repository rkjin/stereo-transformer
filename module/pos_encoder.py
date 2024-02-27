#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import math

import torch
from torch import nn

from utilities.misc import NestedTensor


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs: NestedTensor):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        print("$$$$$$$$$$$ position encoding")
        x = inputs.left
        # update h and w if downsampling
        bs, _, h, w = x.size() # torch.Size([1, 3, 209, 943])
        print('x,h,w,  inputs.sampled_cols, inputs.sampled_rows',x.shape, h, w)
        if inputs.sampled_cols is not None: # input 따라서 바뀜
            bs, w = inputs.sampled_cols.size()
        if inputs.sampled_rows is not None:
            _, h = inputs.sampled_rows.size()
        print('w, h', w, h) #314 70 
        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)
        # scale distance if there is down sample
        if inputs.sampled_cols is not None:
            scale = x.size(-1) / float(inputs.sampled_cols.size(-1)) # 943 / 314 = 3.0031847133757963
            x_embed = x_embed * scale
        if self.normalize: # false
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)#(128)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        print(pos_x.shape, x_embed.shape, dim_t.shape) #torch.Size([627, 128]) torch.Size([627]) torch.Size([128])
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC torch.Size([627, 128])
        print('pos', pos.shape )
        return pos


def no_pos_encoding(x):
    return None


def build_position_encoding(args):

    mode = args.position_encoding # sine1d_rel
    channel_dim = args.channel_dim # 128
    print('mode, channel_dim', mode,  channel_dim)
    if mode == 'sine1d_rel': #true
        n_steps = channel_dim
        position_encoding = PositionEncodingSine1DRelative(n_steps, normalize=False)
    elif mode == 'none':
        position_encoding = no_pos_encoding
    else:
        raise ValueError(f"not supported {mode}")

    return position_encoding
