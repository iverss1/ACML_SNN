#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

from functools import partial
from typing import Tuple

import torch
import math
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

from torch import nn, _assert, Tensor

from utils.helpers import to_2tuple
import torch.nn.functional as F


class ActFun(torch.autograd.Function):

    @staticmethod
    #    def forward(ctx, input):
    #       ctx.save_for_backward(input)
    #        input[torch.where(input.lt(thresh))]=0
    #        #input.float()
    #        out = input
    #        #print(input,"input")
    #       return out

    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


decay = 0.4
thresh = 0.5
lens = 0.5
act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    I, _ = ops(x)
    #print(I.shape,"I")
    #print(spike.shape,"spike")
    #print(mem.shape,"mem")
    #print(mem.max(),"mem()")
    mem = mem * decay * (1. - spike) + I
    #print(I.max(),"I.max()")
    
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


class Surrogate_BP_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, input):
        # print("input1",input)
        self.save_for_backward(input)
        # print("input2",input)
        # out = torch.zeros_like(input).cuda()
        # out[input > 0] = 1.0
        output = input.ge(0).type(torch.cuda.FloatTensor)

        return output

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

count = 0
indicator = 0
mem_temp_g1=None
mem_temp_g2=None
mem_temp_g3=None
mem_temp_g4=None
mem_tempg=None


class SNNBase(nn.Module):
    def __init__(self, index, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, leak_mem=1.0, default_threshold=1.0):
        super(SNNBase, self).__init__()
        self.index = index
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size if bidirectional else hidden_size
        self.union = union
        self.with_vertical = True
        self.with_horizontal = True
        self.with_group = True
        self.with_fc = with_fc
        self.down = PatchMerging(input_size*2)
        affine_flag = True
        bias_flag = True
        if with_fc:
            if union == "cat":
                if index == 0:
                    self.fc = nn.Linear(self.output_size * 2, hidden_size)
                    print(self.output_size, "self.output_size")
                else:
                    self.fc = nn.Linear(self.output_size * 2, hidden_size)
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
        self.snn_g1 = SNNIdentity()
        # self.snn_g2 = SNNIdentity()
        # self.snn_g3 = SNNIdentity()
        # self.snn_g4 = SNNIdentity()
        self.snn_v = SNNIdentity()
        self.snn_h = SNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        global count,indicator,mem_temp_g1,mem_temp_g2,mem_temp_g3,mem_temp_g4,mem_tempg
        
        self.c2_mem = torch.zeros(x.shape[0], 1, self.hidden_size * 2).cuda()
        # print(x.shape,"x_in_snn")
        if self.with_group:
            # print("with_group")
            
            # print("with_patch")
            if indicator % 2 == 0:
                if self.index == 0:     
                    mem_tempg = torch.zeros(x.shape[0], x.shape[1],x.shape[2], self.hidden_size * 2).cuda()        
                B, H, W, C = x.shape
                window_size = int(x.shape[1] / 2)
                #print("window_size",window_size)
                # x=x.reshape(B, window_size**2, C)
                patch_group_1 = x[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                patch_group_2 = x[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                patch_group_3 = x[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                patch_group_4 = x[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                mem_temp_g1 = mem_tempg[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                mem_temp_g2 = mem_tempg[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                mem_temp_g3 = mem_tempg[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                mem_temp_g4 = mem_tempg[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size,-1)
                
                g1, mem_temp_g1, self.c2_mem = self.snn_g1(patch_group_1, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                #print(mem_temp_g1[0, 0, 0],"mem_temp_g1_1")
                
                g2, mem_temp_g2, self.c2_mem = self.snn_g1(patch_group_2, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                #print(mem_temp_g2,"mem_temp_g2")
                g3, mem_temp_g3, self.c2_mem = self.snn_g1(patch_group_3, 2*torch.sin((mem_temp_g3*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                #print(mem_temp_g3,"mem_temp_g3")
                g4, mem_temp_g4, self.c2_mem = self.snn_g1(patch_group_4, 2*torch.sin((mem_temp_g4*math.pi/4)*torch.cos((mem_temp_g3*math.pi/4))), self.c2_mem)
                #print(mem_temp_g4,"mem_temp_g4")
                g1,mem_temp_g1 = g1.reshape(B, window_size, window_size, -1), mem_temp_g1.reshape(B, window_size, window_size, -1)
                
                g2,mem_temp_g2 = g2.reshape(B, window_size, window_size, -1), mem_temp_g2.reshape(B, window_size, window_size, -1)
                g3,mem_temp_g3 = g3.reshape(B, window_size, window_size, -1), mem_temp_g3.reshape(B, window_size, window_size, -1)
                g4,mem_temp_g4 = g4.reshape(B, window_size, window_size, -1), mem_temp_g4.reshape(B, window_size, window_size, -1)

                if self.union == "cat":
                    g_temp1 = torch.cat([g1, g2], dim=2)
                    # print(g_temp1.shape,"g_temp1")
                    g_temp2 = torch.cat([g3, g4], dim=2)
                    g = torch.cat([g_temp1, g_temp2], dim=1)
                    x = g
                    
                    mem_temp1 = torch.cat([mem_temp_g1, mem_temp_g2], dim=2)
                    # print(g_temp1.shape,"g_temp1")
                    mem_temp2 = torch.cat([mem_temp_g3, mem_temp_g4], dim=2)
                    mem_tempg = torch.cat([mem_temp1, mem_temp2], dim=1)
                    indicator += 1
                    # print(x.shape,"x_cat")
                if self.with_fc:
                    #print(x.shape,"x_fc_in_average1")
                    x = self.fc(x)
                    # x=x
                    #print(x.shape,"x_fc_out_averagewindow")
            else:
                if self.index == 3:
                    B, H, W, C = x.shape
                    window_size = int(x.shape[1] / 2)
                    #print("window_size",window_size)
                    # x=x.reshape(B, window_size**2, C)
                    patch_group_1 = x[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    patch_group_2 = x[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                    patch_group_3 = x[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    patch_group_4 = x[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size,-1)
                    #print(patch_group_4.shape,"patch_group_4")
                    mem_temp_g1 = mem_tempg[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g2 = mem_tempg[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g3 = mem_tempg[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g4 = mem_tempg[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size,-1)
                    
                
                    g1, mem_temp_g1, self.c2_mem = self.snn_g1(patch_group_1, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g1[0, 0, 0],"mem_temp_g1_1")
                    
                    g2, mem_temp_g2, self.c2_mem = self.snn_g1(patch_group_2, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g2,"mem_temp_g2")
                    g3, mem_temp_g3, self.c2_mem = self.snn_g1(patch_group_3, 2*torch.sin((mem_temp_g3*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g3,"mem_temp_g3")
                    g4, mem_temp_g4, self.c2_mem = self.snn_g1(patch_group_4, 2*torch.sin((mem_temp_g4*math.pi/4)*torch.cos((mem_temp_g3*math.pi/4))), self.c2_mem)
                    # print(self.c1_mem,"self.c1_mem4")
                    g1,mem_temp_g1 = g1.reshape(B, window_size, window_size, -1), mem_temp_g1.reshape(B, window_size, window_size, -1)
                    g2,mem_temp_g2 = g2.reshape(B, window_size, window_size, -1), mem_temp_g2.reshape(B, window_size, window_size, -1)
                    g3,mem_temp_g3 = g3.reshape(B, window_size, window_size, -1), mem_temp_g3.reshape(B, window_size, window_size, -1)
                    g4,mem_temp_g4 = g4.reshape(B, window_size, window_size, -1), mem_temp_g4.reshape(B, window_size, window_size, -1)

                    if self.union == "cat":
                        g_temp1 = torch.cat([g1, g2], dim=2)
                        # print(g_temp1.shape,"g_temp1")
                        g_temp2 = torch.cat([g3, g4], dim=2)
                        g = torch.cat([g_temp1, g_temp2], dim=1)
                        x = g
                        mem_temp1 = torch.cat([mem_temp_g1, mem_temp_g2], dim=2)
                        # print(g_temp1.shape,"g_temp1")
                        mem_temp2 = torch.cat([mem_temp_g3, mem_temp_g4], dim=2)
                        mem_tempg = torch.cat([mem_temp1, mem_temp2], dim=1)
                        indicator += 1
                        # print(x.shape,"x_cat")
                    if self.with_fc:
                        #print(x.shape,"x_fc_in_average3")
                        x = self.fc(x)
                        count=0
                        #mem_tempg = self.down(mem_tempg)
                        # x=x
                        #print(x.shape,"x_fc_out_average3")
                else:
                    indicator -= 1
                    B, H, W, C = x.shape
                    window_size = int(x.shape[1] * 3 / 4)
                    over_compensation = 2 * window_size - H
                    #print("window_size",window_size)
                    # print("over_compensation",over_compensation)
                    patch_group_1 = x[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    patch_group_2 = x[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                    patch_group_3 = x[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    patch_group_4 = x[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size,-1)
                    
                    mem_temp_g1 = mem_tempg[:, 0:window_size, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g2 = mem_tempg[:, 0:window_size, W - window_size:W, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g3 = mem_tempg[:, H - window_size:H, 0:window_size, :].reshape(B, window_size * window_size, -1)
                    mem_temp_g4 = mem_tempg[:, H - window_size:H, W - window_size:W, :].reshape(B, window_size * window_size,-1)
                    #print(mem_temp_g1[0, 0, 0],"mem_temp_g1_3")
                
                    g1, mem_temp_g1, self.c2_mem = self.snn_g1(patch_group_1, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g1[0, 0, 0],"mem_temp_g1_1")
                    
                    g2, mem_temp_g2, self.c2_mem = self.snn_g1(patch_group_2, 2*torch.sin((mem_temp_g1*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g2,"mem_temp_g2")
                    g3, mem_temp_g3, self.c2_mem = self.snn_g1(patch_group_3, 2*torch.sin((mem_temp_g3*math.pi/4)*torch.cos((mem_temp_g2*math.pi/4))), self.c2_mem)
                    #print(mem_temp_g3,"mem_temp_g3")
                    g4, mem_temp_g4, self.c2_mem = self.snn_g1(patch_group_4, 2*torch.sin((mem_temp_g4*math.pi/4)*torch.cos((mem_temp_g3*math.pi/4))), self.c2_mem)
                    # print(self.c1_mem,"self.c1_mem4")
                    g1,mem_temp_g1 = g1.reshape(B, window_size, window_size, -1), mem_temp_g1.reshape(B, window_size, window_size, -1)
                    g2,mem_temp_g2 = g2.reshape(B, window_size, window_size, -1), mem_temp_g2.reshape(B, window_size, window_size, -1)
                    g3,mem_temp_g3 = g3.reshape(B, window_size, window_size, -1), mem_temp_g3.reshape(B, window_size, window_size, -1)
                    g4,mem_temp_g4 = g4.reshape(B, window_size, window_size, -1), mem_temp_g4.reshape(B, window_size, window_size, -1)
                    if self.union == "cat":
                        mean_g1g2 = (g1[:, :, int(over_compensation / 2):int(over_compensation * 3 / 2), :] + g2[:, :,0:over_compensation,:]) / 2
                        g1_g2 = torch.cat([g1[:, :, 0:int(over_compensation / 2), :], mean_g1g2,g2[:, :, over_compensation:int(over_compensation * 3 / 2), :]], dim=2)
                        mean_g3g4 = (g3[:, :, int(over_compensation / 2):int(over_compensation * 3 / 2), :] + g4[:, :,0:over_compensation,:]) / 2
                        g3_g4 = torch.cat([g3[:, :, 0:int(over_compensation / 2), :], mean_g3g4,g4[:, :, over_compensation:int(over_compensation * 3 / 2), :]], dim=2)
                        mean_g = (g1_g2[:, int(over_compensation / 2):int(over_compensation * 3 / 2), :, :] + g3_g4[:,0:over_compensation,:, :]) / 2
                        g = torch.cat([g1_g2[:, 0:int(over_compensation / 2), :, :], mean_g,g3_g4[:, over_compensation:int(over_compensation * 3 / 2), :, :]], dim=1)
                        x = g
                        
                        mean_mem_g1g2 = (mem_temp_g1[:, :, int(over_compensation / 2):int(over_compensation * 3 / 2), :] + mem_temp_g2[:, :,0:over_compensation,:]) / 2
                        mem_g1_g2 = torch.cat([mem_temp_g1[:, :, 0:int(over_compensation / 2), :], mean_mem_g1g2,mem_temp_g2[:, :, over_compensation:int(over_compensation * 3 / 2), :]], dim=2)
                        mean_mem_g3g4 = (mem_temp_g3[:, :, int(over_compensation / 2):int(over_compensation * 3 / 2), :] + mem_temp_g4[:, :,0:over_compensation,:]) / 2
                        mem_g3_g4 = torch.cat([mem_temp_g3[:, :, 0:int(over_compensation / 2), :], mean_mem_g3g4,mem_temp_g4[:, :, over_compensation:int(over_compensation * 3 / 2), :]], dim=2)
                        mean_mem_g = (mem_g1_g2[:, int(over_compensation / 2):int(over_compensation * 3 / 2), :, :] + mem_g3_g4[:,0:over_compensation,:, :]) / 2
                        mem_tempg = torch.cat([mem_g1_g2[:, 0:int(over_compensation / 2), :, :], mean_mem_g,mem_g3_g4[:, over_compensation:int(over_compensation * 3 / 2), :, :]], dim=1)
                        
                    if self.with_fc:
                        #print(x.shape,"x_fc_in_overl")
                        x = self.fc(x)
                        if self.index == 0 and count==0:
                            count=0
                            mem_tempg = self.down(mem_tempg)                        
                        elif self.index == 1 and count==0:
                            #print("down1")
                            count=0
                            mem_tempg = self.down(mem_tempg)
                            #print(mem_tempg.shape,"mem_tempg")
                        
                        elif self.index == 2 and count==2:
                            #print("down2")
                            count=0
                            mem_tempg = self.down(mem_tempg)
                            
                            #print(mem_tempg.shape,"mem_tempg")
                        else:
                            count+=1
                        # x=x
                        #print(x.shape,"x_fc_out_overl")

        return x


class SNN2D(SNNBase):
    def __init__(self, index, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, leak_mem=1.0, default_threshold=1.0):
        super().__init__(index, input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc, leak_mem,
                         default_threshold)
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.threshold = default_threshold
        self.union = union
        self.with_fc = with_fc
        if self.with_group:
            self.snn_g1 = BiSNN(index, input_size, hidden_size, bidirectional, leak_mem=leak_mem,
                                default_threshold=default_threshold)
            # self.snn_g2 = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
            #                  default_threshold=default_threshold)
            # self.snn_g3 = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
            #                   default_threshold=default_threshold)
            # self.snn_g4 = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
            #                   default_threshold=default_threshold)
        # if self.with_vertical:
        #    self.snn_v = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
        #                       default_threshold=default_threshold)
        # if self.with_horizontal:
        #    self.snn_h = BiSNN(index,input_size, hidden_size, bidirectional, leak_mem=leak_mem,
        #                       default_threshold=default_threshold)
        print("SNN2D APPLY")


class ScNN(nn.Module):
    def __init__(self, index, input_size: int, hidden_size: int, leak_mem=1.0,
                 default_threshold=1.0):
        super(SNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.conv1 = nn.Conv2d(self.input_size, self.hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)

        # self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        # self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, x, c1_mem, c2_mem):
        c1_spike = torch.zeros(x.shape[0], self.hidden_size, 1, 1).cuda()
        c2_spike = torch.zeros(x.shape[0], self.hidden_size, 1, 1).cuda()
        time_step = 1
        for step in range(time_step):  # simulation time steps
            x = x > torch.rand(x.size()).cuda()  # prob. firing
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            c2_mem, c2_spike = mem_update(self.conv2,c1_spike, c2_mem,c2_spike)

        outputs = c2_spike.transpose(1, 3)
        # print(outputs.shape,"outputs")
        return outputs, c1_mem, c2_mem


class SNN(nn.Module):
    def __init__(self, index, input_size: int, hidden_size: int, leak_mem=1.0,
                 default_threshold=1.0):
        super(SNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bias=True,
                             bidirectional=True)
        # self.lstm2 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bias=True,
        #                         bidirectional=True)

        # self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        # self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, x, c1_mem, c2_mem):
        c1_spike = torch.zeros(x.shape[0], 1, self.hidden_size * 2).cuda()
        #print(x.shape[1],"timestep")
        time_step = 1
        for step in range(time_step):  # simulation time steps
            x = x > torch.rand(x.size()).cuda()  # prob. firing
            c1_mem, c1_spike = mem_update(self.lstm1, x.float(), c1_mem, c1_spike)
            # c2_mem, c2_spike = mem_update(self.lstm2,c1_spike, c2_mem,c2_spike)

        outputs = c1_spike
        # print(outputs.shape,"outputs")
        return outputs, c1_mem, c2_mem


class BiSNN(nn.Module):
    def __init__(self, index, input_size: int, hidden_size: int, bidirectional: bool, leak_mem=1.0,
                 default_threshold=1.0):
        super(BiSNN, self).__init__()
        self.bidrectional = bidirectional
        self.forward_snn = SNN(index, input_size, hidden_size, leak_mem=leak_mem,
                               default_threshold=default_threshold)
        #self.backward_snn = SNN(index, input_size, hidden_size, leak_mem=leak_mem,
        #                        default_threshold=default_threshold)
        #self.conv = nn.Conv1d(input_size, hidden_size * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, m1, m2):

        if self.bidrectional:
            print("bi-directional")
            forward_out = self.forward_snn(x)
            #backward_out = self.backward_snn(torch.flip(x, [1]))
            backward_out = torch.flip(backward_out, [1])
            x = torch.cat([forward_out, backward_out], dim=-1)
            return x
        else:
            # print("directional")
            x, m1, m2 = self.forward_snn(x, m1, m2)
            # x = self.conv(x)
            return x, m1, m2


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
            # print(x.shape, "rnn_v_1")
            v, _ = self.rnn_v(v)
            # print(x.shape, "rnn_v_1")
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
    def __init__(self, index, dim, hidden_size, mlp_ratio=3.0, rnn_layer=SNN2D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        dim = dim
        if index == 3:
            dim1 = dim
        else:
            dim1 = dim
        #self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(index, dim1, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    union=union, with_fc=with_fc, leak_mem=1.0, default_threshold=1.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(self.rnn_tokens(self.norm1(x)).shape,"self.rnn_tokens(self.norm1(x))")
        # print(x.shape,"x")
        x = x + self.drop_path(self.rnn_tokens(x))
        #print(x,"x")
        x = x + self.drop_path(self.mlp_channels(x))
        #print(x.shape,"x_forward")
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

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        # 确实是需要padding的，前面被4整除后不一定能再以2降了
        H = x.shape[1]
        W = x.shape[2]
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 这个地方的意思是 如x0,B维度不处理,行上以0为起点,以2为步距进行取值,C维度不处理
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        #x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        #x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

class Downsample2D(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv = nn.Conv2d(input_dim,output_dim,kernel_size = 1,stride = 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        # print(x.shape,"downsample")
        return x
