# C++/CUDA Extensions in Pytorch by Gwanjyun
一般来说，要实现上述功能要进行三步骤
 - 编写纯python的pytorch类，包括前向和后向操作，确保原始代码(主要是后向传播)的正确性。先写经典pytorch代码，然后编写前向后向分离的版本。通过两个版本进行前向和后向梯度结果对比。结果保存在python文件夹下。
 - 编写c++代码
 - 编写c++/cuda代码

## LLTM的重要的数学原理（Demo实例）
$$
x^{n}_{h}, x^{n}_{c} = f_\theta(y^{n}, x^{n-1}_{h}, x^{n-1}_{c})
$$
步骤为:
$$
\begin{align}
    x^{n-1} &= [y^{n},x^{n-1}_{h}] \quad\mathrm{where, [\cdot]\ means\ concat} \\
    g^{n} &= Wx^{n-1} + b \\
    [g_{i}^{n}, g_{o}^{n}, g_{c}^{n}] &= g^{n} \\
    w_{i}^{n} &= Sigmoid(g_{i}^{n}) \\
    w_{o}^{n} &= Sigmoid(g_{o}^{n}) \\
    w_{c}^{n} &= Elu(g_{i}^{n}) \\
    x^{n}_{c} &= x^{n-1}_{c} + w_{c}^{n}\cdot w_{i}^{n} \\
    x^{n}_{h} &= Tanh(x^{n}_{c})*w_{o}^{n}
\end{align}
$$

## 纯python的pytorch类
- [ ] Python基本Pytorch类
- [ ] Pytorch类拆分为forward + backward函数
## 编写c++代码
## 编写c++/cuda代码