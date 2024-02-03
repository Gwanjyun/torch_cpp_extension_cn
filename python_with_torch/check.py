import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
from python.lltm import LLTM as LLTM
from python.lltm_base_line import LLTM as LLTM_base_line
from cpp.lltm_cpp import LLTM as LLTM_cpp
from cuda.lltm_cuda import LLTM as LLTM_cuda

from tqdm import trange, tqdm
import time

torch.set_default_dtype(torch.float64) # 32 for some error but fast, 64 for accurate but slow
def check_equal(first, second, verbose):
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        # if torch.allclose(x, y, rtol=1e-7, atol=1e-7):
        #     print('True')
        # else:
        #     print('False')
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i), rtol=1e-7, atol=1e-7)
        
def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()

'''输入参数'''
torch.manual_seed(21216258) # 随机种子
test_N = 1000
device = torch.device('cuda:1')
# device = torch.device('cpu')
B = 5000
C = 128
C1 = 128
x = torch.randn(B,C, device=device, requires_grad=True)
w = torch.randn(C+C1, C1*3, device=device, requires_grad=True)
b = torch.randn(C1*3, device=device, requires_grad=True)
h = torch.randn(B,C1, device=device, requires_grad=True)
c = torch.randn(B,C1, device=device, requires_grad=True)
state = [h,c]
variables = [w,b,x,h,c]
'''实例化'''
torch.manual_seed(21216258) # 随机种子
lltm = LLTM(C,C1).to(device)
torch.manual_seed(21216258) # 随机种子
lltm_bl = LLTM_base_line(C,C1).to(device)
torch.manual_seed(21216258) # 随机种子
lltm_cpp = LLTM_cpp(C,C1).to(device)
torch.manual_seed(21216258) # 随机种子
lltm_cuda = LLTM_cuda(C,C1).to(device)


'''测试'''
out = lltm(x, state)
(out[0] + out[1]).sum().backward()
grad = [i.grad.clone() for i in lltm.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
zero_grad([i for i in lltm.parameters()] + [x, h, c])


out_bl = lltm_bl(x, state)
(out_bl[0] + out_bl[1]).sum().backward()
grad_bl = [i.grad.clone() for i in lltm_bl.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
zero_grad([i for i in lltm_bl.parameters()] + [x, h, c])
# check_equal(out, out_bl, True)
# check_equal(grad, grad_bl, True)

out_cpp = lltm_cpp(x, state)
(out_cpp[0] + out_cpp[1]).sum().backward()
grad_cpp = [i.grad.clone() for i in lltm_cpp.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
zero_grad([i for i in lltm_cpp.parameters()] + [x, h, c])

out_cuda = lltm_cuda(x, state)
(out_cuda[0] + out_cuda[1]).sum().backward()
grad_cuda = [i.grad.clone() for i in lltm_cuda.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
zero_grad([i for i in lltm_cuda.parameters()] + [x, h, c])


'''pytorch class'''
tic = time.time()
for i in trange(test_N):
    out = lltm(x, state)
    (out[0] + out[1]).sum().backward()
    grad = [i.grad.clone() for i in lltm.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
    zero_grad([i for i in lltm.parameters()] + [x, h, c])
torch.cuda.synchronize(device)
toc = time.time()
print(toc-tic)

'''python with custom'''
tic = time.time()
for i in trange(test_N):
    out_bl = lltm_bl(x, state)
    (out_bl[0] + out_bl[1]).sum().backward()
    grad_bl = [i.grad.clone() for i in lltm_bl.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
    zero_grad([i for i in lltm_bl.parameters()] + [x, h, c])
torch.cuda.synchronize(device)
toc = time.time()
print(toc-tic)

'''cpp in cpu'''
tic = time.time()
for i in trange(test_N):
    out_cpp = lltm_cpp(x, state)
    (out_cpp[0] + out_cpp[1]).sum().backward()
    grad_cpp = [i.grad.clone() for i in lltm_cpp.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
    zero_grad([i for i in lltm_cpp.parameters()] + [x, h, c])
torch.cuda.synchronize(device)
toc = time.time()
print(toc-tic)

'''cuda'''
tic = time.time()
for i in trange(test_N):
    out_cuda = lltm_cuda(x, state)
    (out_cuda[0] + out_cuda[1]).sum().backward()
    grad_cuda = [i.grad.clone() for i in lltm_cuda.parameters()] + [x.grad.clone(), h.grad.clone(), c.grad.clone()]
    zero_grad([i for i in lltm_cuda.parameters()] + [x, h, c])
torch.cuda.synchronize(device)
toc = time.time()
print(toc-tic)

check_equal(out, out_cuda, False)
check_equal(out, out_cpp, False)
check_equal(out, out_bl, False)
check_equal(grad, grad_cuda, False)
check_equal(grad, grad_bl, False)
check_equal(grad, grad_cpp, False)

