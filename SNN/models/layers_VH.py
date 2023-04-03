#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

from functools import partial
from typing import Tuple

import torch
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

from torch import nn, _assert, Tensor

from utils.helpers import to_2tuple
import torch.nn.functional as F

class Surrogate_BP_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        # print("input1",input)
        self.save_for_backward(input)
        # print("input2",input)
        # out = torch.zeros_like(input).cuda()
        # out[input > 0] = 1.0
        return input.ge(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad = torch.exp( -(grad_input - 0.3) **2/(2 * 0.3 ** 2) ) / ((2 * 0.3 * 3.141592653589793) ** 0.5)
        # grad = grad_input * grad
        # grad = abs(grad_input - threshold) < lens
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0) * 2
        # grad=F.hardtanh(grad_input)
        # grad =grad_input * 0.3*torch.exp(-0.01*torch.abs(input))
        # print(grad)
        # grad =grad_input * 0.3 * torch.exp(F.threshold(1.0 - torch.abs(input), 0, 0))
        # hu = abs(input) < aa
        # hu = hu.float() / (2 * aa)
        return grad  # grad_input*hu


class RNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        return x, None


class SNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNNIdentity, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        return x, None


class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__()
        self.rnn = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        x, _ = self.rnn(x.view(B, -1, C))
        return x.view(B, H, W, -1)


class RNN(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class GRU(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional)


class LSTM(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bias=bias, bidirectional=bidirectional)
        print("LSTM Apply")


class SNNBase(nn.Module):
    def __init__(self,index,input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, leak_mem=1.0, default_threshold=1.0):
        super(SNNBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size if bidirectional else hidden_size
        self.union = union
        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        affine_flag = True
        bias_flag = True
        if with_fc:
            if union == "cat":
                if index == 0:
                    
                    self.fc = nn.Linear(self.output_size, 192)
                else:
                    self.fc = nn.Linear(self.output_size*2, 384)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(
                    f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.snn_v = SNNIdentity()
        self.snn_h = SNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(B, H, -1)
            v = self.snn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)

        if self.with_horizontal:
            h = x.reshape(B, W, -1)
            h = self.snn_h(h)
            h = h.reshape(B, H, W, -1)

        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            
            x = self.fc(x)
            
        return x


class SNN2D(SNNBase):
    def __init__(self,index,input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, leak_mem=1.0, default_threshold=1.0):
        super().__init__(index,input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc, leak_mem,
                         default_threshold)
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.threshold = default_threshold
        self.union = union
        self.with_fc = with_fc
        if self.with_vertical:
            self.snn_v = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
                               default_threshold=default_threshold)
        if self.with_horizontal:
            self.snn_h = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
                               default_threshold=default_threshold)

        print("SNN2D APPLY")


class SNN(nn.Module):
    def __init__(self,index,input_size: int, hidden_size: int, leak_mem=1.0,
                 default_threshold=1.0):
        super(SNN, self).__init__()
        self.hidden_size= hidden_size
        print(input_size,hidden_size,"input_size,hidden_size")
        bias_flag = True
        affine_flag = True
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.threshold = default_threshold
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.BatchNorm1d(hidden_size, eps=1e-4, momentum=0.9, affine=affine_flag)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.BatchNorm1d(hidden_size, eps=1e-4, momentum=0.9, affine=affine_flag)

        self.threshold1 = nn.Parameter(torch.tensor(self.threshold))
        self.threshold2 = nn.Parameter(torch.tensor(self.threshold))
        self.threshold3 = nn.Parameter(torch.tensor(self.threshold))

    def forward(self,x):
        time_step = x.shape[1]
        output=torch.zeros(x.shape[0], x.shape[1],self.hidden_size).cuda()
        self.mem_conv1 = torch.zeros(x.shape[0], self.hidden_size, 1).cuda()
        self.mem_conv2 = torch.zeros(x.shape[0], self.hidden_size, 1).cuda()
        for t in range(time_step):
            patch = x[:, t, :].unsqueeze(-1)
            self.mem_conv1 = self.leak_mem * self.mem_conv1 + self.conv1(patch)
            mem_thr = (self.mem_conv1 / self.threshold1 - 1.0)  ## self.mem_conv1 thresold
            out = self.spike_fn(mem_thr)
            rst = self.threshold1 * (mem_thr > 0).float()  # (mem_thr>0) return 1
            mem_conv = self.mem_conv1.clone()
            self.mem_conv1 = mem_conv - rst
            out_prev1 = out.clone()
            self.mem_conv2 = self.leak_mem * self.mem_conv2 + self.conv2(out_prev1)
            mem_thr = (self.mem_conv2 / self.threshold2) - 1.0  ###mem_conv2 compare with threshold
            out = self.spike_fn(mem_thr)
            rst = self.threshold2 * (mem_thr > 0).float()
            self.mem_conv2 = self.mem_conv2 - rst  ###rst=0 means mem_thr<0,means mem_conv2<threshold,thus:mem_conv2 not change
            out_prev2 = out.clone()  ###rst=threshold means mem_thr>0,means mem_conv2>threshold,thus:mem_conv2 =0 or mem_conv2 - threshold
            out_prev2=out_prev2.squeeze(-1)
            #print(out_prev3.shape,"out_prev3")
            output[:, t, :] = out_prev2
        return output


class BiSNN(nn.Module):
    def __init__(self,index,input_size: int, hidden_size: int, bidirectional: bool,leak_mem=1.0,
                 default_threshold=1.0):
        super(BiSNN, self).__init__()
        self.bidrectional = bidirectional
        self.forward_snn = SNN(index,input_size, hidden_size, leak_mem=leak_mem,
                               default_threshold=default_threshold)
        self.backward_snn = SNN(index,input_size, hidden_size, leak_mem=leak_mem,
                                default_threshold=default_threshold)
        self.conv = nn.Conv1d(input_size, hidden_size*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        if self.bidrectional:
            #print("bi-directional")
            forward_out = self.forward_snn(x)
            backward_out = self.backward_snn(torch.flip(x, [1]))
            backward_out = torch.flip(backward_out, [1])
            x = torch.cat([forward_out, backward_out], dim=-1)
            return x
        else:
            #print("directional")
            x = self.forward_snn(x)
            x = self.conv(x)
            return x



class RNN2DBase(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(
                    f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(
                    f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            print(x.shape, "rnn_v_1")
            v, _ = self.rnn_v(v)
            print(x.shape, "rnn_v_1")
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)

        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)

        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)
        return x


class RNN2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_horizontal:
            self.rnn_h = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                bidirectional=bidirectional, nonlinearity=nonlinearity)


class LSTM2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                 bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                 bidirectional=bidirectional)


class GRU2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias,
                                bidirectional=bidirectional)


class VanillaSequencerBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Sequencer2DBlock(nn.Module):
    def __init__(self, index,dim,hidden_size, mlp_ratio=3.0, rnn_layer=SNN2D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        dim = dim
        if index == 0:
            dim1 = dim*4
        else:
            dim1 = dim*2
        #self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(index,dim1, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    union=union, with_fc=with_fc, leak_mem=1.0, default_threshold=1.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #print(self.rnn_tokens(self.norm1(x)).shape,"self.rnn_tokens(self.norm1(x))")
        x = x + self.drop_path(self.rnn_tokens(x))
        x = x + self.drop_path(self.mlp_channels(x))
        return x


class PatchEmbed(TimmPatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2D(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x
