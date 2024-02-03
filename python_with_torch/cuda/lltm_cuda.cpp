#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations
std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor old_c
);
std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor weights,
    torch::Tensor X,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor gate_weights,
    torch::Tensor new_cell
);

// C++ interface
std::vector<torch::Tensor> lltm_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor old_c){
    // CHECK is neccesary.
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(input);
    CHECK_INPUT(old_h);
    CHECK_INPUT(old_c);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input)); //不同GPU
    return lltm_cuda_forward(weights, bias, input, old_h, old_c);
}


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
    // CHECK is neccesary.
    CHECK_INPUT(grad_h);
    CHECK_INPUT(grad_cell);
    CHECK_INPUT(weights);
    CHECK_INPUT(X);
    CHECK_INPUT(input_gate);
    CHECK_INPUT(output_gate);
    CHECK_INPUT(candidate_cell);
    CHECK_INPUT(gate_weights);
    CHECK_INPUT(new_cell);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_h)); //不同GPU
    return lltm_cuda_backward(
        grad_h,
        grad_cell,
        weights,
        X,
        input_gate,
        output_gate,
        candidate_cell,
        gate_weights,
        new_cell
    );
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}



