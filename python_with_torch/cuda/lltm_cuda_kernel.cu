#include <torch/extension.h>
// #include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__device__ scalar_t sigmoid(scalar_t x){
    const auto y = 1.0 / (1.0 + exp(-x));
    return y;
}

template <typename scalar_t>
__device__ scalar_t d_sigmoid(scalar_t x){
    const auto y = sigmoid(x);
    return (1.0 - y)*y;
}


template <typename scalar_t>
__device__ scalar_t tanh(scalar_t x){
    const auto exp0 = exp(-x);
    const auto exp1 = exp(x);
    return (exp1 - exp0) / (exp1 + exp0);
}

template <typename scalar_t>
__device__ scalar_t d_tanh(scalar_t x){
    const auto y = tanh(x);
    return 1.0 - y*y;
}

template <typename scalar_t>
__device__ scalar_t elu(scalar_t x, scalar_t alpha=1.0){
    if(x>=0){
        return x;
    }
    else{
        return alpha*(exp(x)-1);
    }
}

template <typename scalar_t>
__device__ scalar_t d_elu(scalar_t x, scalar_t alpha=1.0){
    const auto exp0 = exp(x);
    const auto d_elu = x >= 0 ? 1 : alpha*exp0;
    return d_elu; 
}

template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_c,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell
){
    const int b = blockIdx.y * blockDim.y + threadIdx.y; //batch
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // channel
    
    if(c >= gates.size(2) || b >= gates.size(0)) return;
    input_gate[b][c] = sigmoid(gates[b][0][c]);
    output_gate[b][c] = sigmoid(gates[b][1][c]);
    candidate_cell[b][c] = elu(gates[b][2][c]);

    new_cell[b][c] = old_c[b][c] + candidate_cell[b][c]*input_gate[b][c];

    new_h[b][c] = tanh(new_cell[b][c]) * output_gate[b][c];

}


std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor input,
    torch::Tensor old_h,
    torch::Tensor old_c
){
    // 不需要并行运算的操作
    auto X = torch::cat({old_h, input}, /*dim=*/1);
    auto gate_weights = torch::mm(X, weights) + bias;
    
    const auto batch_size = old_h.size(0); // 开辟的空间
    const auto state_size = old_h.size(1); // 开辟的空间
    auto gates = gate_weights.reshape({batch_size, 3, state_size});
    // 并行输出的变量
    auto myoptions = torch::dtype(weights.dtype()).device(weights.device());
    auto new_h = torch::zeros({batch_size, state_size}, myoptions);
    auto new_cell = torch::zeros({batch_size, state_size}, myoptions);
    auto input_gate = torch::zeros({batch_size, state_size}, myoptions);
    auto output_gate = torch::zeros({batch_size, state_size}, myoptions);
    auto candidate_cell = torch::zeros({batch_size, state_size}, myoptions);


    // // 开辟线程数
    const dim3 threads(4, 256); //修改这个可以有不同加速效果
    const dim3 blocks((state_size + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);

    //经典操作
    AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_cuda_forward", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        // 输入
        old_c.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        //输出
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));

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

template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    // 输入为const
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_cell,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> gate_weights,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell,
    // 输出为非const
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_old_c,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_gate_weights
){
    // 当前kernel的索引
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;

    const int batch_size = grad_h.size(0);
    const int state_size = grad_h.size(1);

    if( b>=batch_size || c >= state_size) return;
    // -1
    auto d_output_gate = grad_h[b][c] * tanh(new_cell[b][c]);
    auto d_tanh_new_cell = grad_h[b][c] * output_gate[b][c];
    auto d_new_cell = d_tanh_new_cell * d_tanh(new_cell[b][c]) + grad_cell[b][c];
    // -2
    d_old_c[b][c] = d_new_cell;
    auto d_candidate_cell = d_new_cell * input_gate[b][c];
    auto d_input_gate = d_new_cell * candidate_cell[b][c];
    // -3~-5
    d_candidate_cell = d_candidate_cell * d_elu(gate_weights[b][c + state_size*2]);
    d_output_gate = d_output_gate * d_sigmoid(gate_weights[b][c + state_size]);
    d_input_gate = d_input_gate * d_sigmoid(gate_weights[b][c]);
    // -6
    d_gate_weights[b][c] = d_input_gate;
    d_gate_weights[b][c + state_size] = d_output_gate;
    d_gate_weights[b][c + state_size*2] = d_candidate_cell;
    // -7
    // -8
}

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
){
    //定义输出变量
    // -1~-6
    auto d_old_c = torch::zeros_like(grad_cell, grad_cell.options());
    auto d_gate_weights = torch::zeros_like(gate_weights, gate_weights.options());

    //分为两维度并行：batch维度 + channel维度
    const int batch_size = new_cell.size(0);
    const int state_size = new_cell.size(1);

    const dim3 threads(4, 256);
    // const dim3 blocks((state_size + threads.x - 1) / threads.x, batch_size);
    const dim3 blocks((state_size + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);

    //经典操作
    AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_cuda_backward", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        // 输入
        grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gate_weights.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        //输出
        d_old_c.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_gate_weights.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
    }));


    // -7
    auto d_weights = torch::mm(X.t(), d_gate_weights);
    auto d_bias = d_gate_weights.sum(0);
    auto d_X = torch::mm(d_gate_weights, weights.t());
    // -8
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