import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(21216258) # 随机种子

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
        old_h, old_c = state
        X = torch.cat([old_h, input], dim=1) # B, C+C1
        gate_weights = X@self.weights + self.bias
        gates = gate_weights.chunk(3, dim=1) # B, 3C1 -> [(B,C1), (B,C1), (B,C1)]
        
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])
        
        # Compute the new cell state.
        new_cell = old_c + candidate_cell * input_gate

        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate
        
        return new_h, new_cell
    

if __name__ == '__main__':
    device = torch.device('cuda:0')
    B = 32
    C = 64
    C1 = 48
    x = torch.randn(B,C, device=device)
    h = torch.randn(B,C1, device=device)
    c = torch.randn(B,C1, device=device)
    state = [h,c]
    model = LLTM(C,C1).to(device)
    
    print(model(x, state))

