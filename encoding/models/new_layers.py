import torch
import torch.nn as nn
import numpy as np 
from torch.autograd.function import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Q_A(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)    
        return x.sign()                     
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input>1.0, 0.0)
        grad_input.masked_fill_(input<-1.0, 0.0)
        mask_pos = (input>=0.0) & (input<1.0)
        mask_neg = (input<0.0) & (input>=-1.0)
        grad_input.masked_scatter_(mask_pos, input[mask_pos].mul_(-2.0).add_(2.0)) 
        grad_input.masked_scatter_(mask_neg, input[mask_neg].mul_(2.0).add_(2.0)) 
        return grad_input * grad_output



class Q_A_DOREFA(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, x):
        return torch.round(nonlinear(x))
    @staticmethod
    def backward(ctx, g):
        return g, None



class Q_W(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x):
        return x.sign() * x.abs().mean()
    @staticmethod
    def backward(ctx, grad):
        return grad


def quantize_a(x):
    x = Q_A.apply(x)
    return x


def quantize_w(x):
    x = Q_W.apply(x)
    return x


def fw(x, bitW):
    if bitW == 32:
        return x
    x = quantize_w(x)
    return x


def fa(x, bitA):
    if bitA == 32:
        return x
    return quantize_a(x)


def nonlinear(x):
    return torch.clamp(torch.clamp(x, max=1.0), min=0.0)



class self_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(self_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)  
            input = F.pad(input, padding_shape, 'constant', 1)       #padding 1
        output = F.conv2d(input, quantize_w(self.weight), bias=self.bias, stride=self.stride, dilation=self.dilation, groups=self.groups)
        return output



class self_conv_zero_padding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(self_conv_zero_padding, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        output = F.conv2d(input, quantize_w(self.weight), bias=self.bias, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
        return output



class group_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_bases=5, stride=1, padding=0, dilation=1, bias=False):
        super(group_conv, self).__init__()

        self.padding = padding
        self.num_bases = num_bases
        self.convs = nn.ModuleList([self_conv_zero_padding(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias) for i in range(num_bases)])

    def forward(self, x):
        output = None
        x = Q_A_DOREFA.apply(x)

        for module in self.convs:
            if output is None:
                output = module(x)
            else:
                output += module(x)
        return output / self.num_bases  



class clip_nonlinear(nn.Module):
    def __init__(self, bitA):
        super(clip_nonlinear, self).__init__()
        self.bitA = bitA

    def forward(self, input):
        return fa(input, self.bitA)


