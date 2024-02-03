#include <torch/extension.h>
#include <iostream>
#include <vector>

// Just do it as what "lltm_base_line.py" has done.

torch::Tensor d_sigmoid(torch::Tensor x){
    auto y = torch::sigmoid(x);
    return (1-y) * y;
}

torch::Tensor d_tanh(torch::Tensor x){
    auto y = torch::tanh(x);
    return 1 - y * y;
}

torch::Tensor d_elu(torch::Tensor x, torch::Scalar alpha=1.0){
    auto mask = x >= 0;
    mask = mask.type_as(x);
    auto exp = torch::exp(x*(1-mask));
    auto out = alpha*exp*(1-mask) + mask;
    return out;
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor old_c){
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gate_weights = torch::mm(X, weights) + bias;
    auto gates = gate_weights.chunk(3, /*dim=*/1);

    auto input_gate = torch::sigmoid(gates[0]);
    auto output_gate = torch::sigmoid(gates[1]);
    auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

    auto new_cell = old_c + candidate_cell * input_gate;

    auto new_h = torch::tanh(new_cell) * output_gate;
    return {
        new_h,
        new_cell,
        X,
        gate_weights,
        input_gate, 
        output_gate,
        candidate_cell
    };
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor weights,
    torch::Tensor X,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor gate_weights,
    torch::Tensor new_cell){
    // -1
    auto d_output_gate = grad_h * torch::tanh(new_cell);
    auto d_tanh_new_cell = grad_h * output_gate;
    auto d_new_cell = d_tanh_new_cell * d_tanh(new_cell) + grad_cell;
    // -2
    auto d_old_c = d_new_cell * 1;
    auto d_candidate_cell = d_new_cell * input_gate;
    auto d_input_gate = d_new_cell * candidate_cell;
    // -3~-5
    auto gates = gate_weights.chunk(3, /*dim*/1);
    d_candidate_cell = d_candidate_cell * d_elu(gates[2]);
    d_output_gate = d_output_gate * d_sigmoid(gates[1]);
    d_input_gate = d_input_gate * d_sigmoid(gates[0]);
    // -6
    auto d_gate_weights = torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim*/1);
    // -7
    auto d_weights = torch::mm(X.t(), d_gate_weights);
    auto d_bias = d_gate_weights.sum(0);
    auto d_X = torch::mm(d_gate_weights, weights.t());
    // -8
    const auto state_size = grad_cell.size(1);
    auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
    auto d_input = d_X.slice(/*dim=*/1, state_size);

    return {
        d_weights,
        d_bias,
        d_input,
        d_old_h,
        d_old_c
    };
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}



