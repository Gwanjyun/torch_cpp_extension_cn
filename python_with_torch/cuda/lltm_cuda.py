import math
from torch import nn
from torch.autograd import Function
import torch

import gwanjyun_lltm_cpp
import gwanjyun_lltm_cuda

torch.manual_seed(21216258) # 随机种子
'''偏微分函数'''
def d_sigmoid(x):
    y = torch.sigmoid(x)
    return (1-y)*y

def d_tanh(x):
    y = torch.tanh(x)
    return 1 - y*y

def d_elu(x, alpha=1.0):
    # x_in = torch.clamp(x, max=0)
    # y = torch.exp(x_in)
    # y[x_in<0] = alpha*y[x_in<0]
    mask = x >= 0
    mask = mask.type_as(x)
    exp = torch.exp(x*(1-mask))
    out = alpha*exp*(1-mask) + mask
    return out

class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = nn.Parameter(
            torch.randn(input_features + state_size, 3*state_size)
        )
        self.bias = nn.Parameter(
            torch.randn(3*state_size)
        )
        
    def forward(self, input, state):
        '''
        input: B,C
        state: [(B,C1), (B,C1)]
        '''
        return LLTMFunction.apply(self.weights, self.bias, input, *state) # 直接调用

class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, weights, bias, input, old_h, old_c):
        outputs = gwanjyun_lltm_cuda.forward(weights.contiguous(), bias.contiguous(), input.contiguous(), old_h.contiguous(), old_c.contiguous())
        new_h, new_cell, X, gate_weights, input_gate, output_gate, candidate_cell = outputs
        
        # 保存计算梯度所需要的变量
        ctx.save_for_backward(weights, X, input_gate, output_gate, candidate_cell, gate_weights, new_cell)
        
        return new_h, new_cell
    
    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True) # for debug
        weights, X, input_gate, output_gate, candidate_cell, gate_weights, new_cell = ctx.saved_variables
        outputs = gwanjyun_lltm_cuda.backward(
            grad_h.contiguous(), 
            grad_cell.contiguous(), 
            weights.contiguous(), 
            X.contiguous(), 
            input_gate.contiguous(), 
            output_gate.contiguous(), 
            candidate_cell.contiguous(), 
            gate_weights.contiguous(), 
            new_cell.contiguous()
        )
        d_weights, d_bias, d_input, d_old_h, d_old_c = outputs
        return d_weights, d_bias, d_input, d_old_h, d_old_c