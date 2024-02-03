from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function # class as Function

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
        X = torch.cat([old_h, input], dim=1) # B, C+C1
        gate_weights = X@weights + bias
        gates = gate_weights.chunk(3, dim=1) # B, 3C1 -> [(B,C1), (B,C1), (B,C1)]
        
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])
        
        # Compute the new cell state.
        new_cell = old_c + candidate_cell * input_gate
        
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate
        
        # 保存计算梯度所需要的变量
        ctx.save_for_backward(X, weights, input_gate, output_gate, new_cell, candidate_cell, gate_weights)
        
        return new_h, new_cell
    
    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True) # for debug
        X, weights, input_gate, output_gate, new_cell, candidate_cell, gate_weights = ctx.saved_variables
        d_weights = d_bias = d_input = d_old_h = d_old_c = None # 输出需要的变量，对应于输入
        '''-1'''
        d_output_gate = grad_h * torch.tanh(new_cell)
        d_tanh_new_cell = grad_h * output_gate
        d_new_cell = d_tanh_new_cell * d_tanh(new_cell) + grad_cell
        '''-2'''
        d_old_c = d_new_cell * 1
        d_candidate_cell = d_new_cell * input_gate
        d_input_gate = d_new_cell * candidate_cell
        '''-3~-5'''
        gates = gate_weights.chunk(3, dim=1)
        d_candidate_cell = d_candidate_cell * d_elu(gates[2])
        d_output_gate = d_output_gate * d_sigmoid(gates[1])
        d_input_gate = d_input_gate * d_sigmoid(gates[0])
        '''-6'''
        d_gate_weights = torch.cat([d_input_gate, d_output_gate, d_candidate_cell], dim=1) # B,3C1
        '''-7'''
        d_weights = X.T@d_gate_weights                                 # (C+C1),3C1
        d_bias = d_gate_weights.sum(dim=0)
        d_X = d_gate_weights@weights.T                          # B,(C+C1)
        '''-8'''
        state_size = grad_cell.shape[1]
        d_old_h, d_input = d_X[:, :state_size], d_X[:, state_size:]
        
        return d_weights, d_bias, d_input, d_old_h, d_old_c
        
        
if __name__ == '__main__':
    device = torch.device('cuda:0')
    B = 32
    C = 64
    C1 = 32
    x = torch.randn(B,C, device=device)
    h = torch.randn(B,C1, device=device)
    c = torch.randn(B,C1, device=device)
    state = [h,c]
    model = LLTM(C,C1).to(device)
    
    print(model(x, state))
        
        
        
        
    


    