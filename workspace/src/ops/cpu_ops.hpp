#pragma once

#include "opendll/tensor.hpp"

namespace opendll::cpu_ops {

// CPU 参考实现：通过抽象接口 download/upload，用 C++ 循环计算。
// 作为数值正确性的黄金基准，供 OpenGL 后端 diff 对比。

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b);
Tensor relu(Device& dev, const Tensor& x);
Tensor add(Device& dev, const Tensor& a, const Tensor& b);
Tensor mul(Device& dev, const Tensor& a, const Tensor& b);
Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding);
Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b);
Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);
Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);
Tensor avgpool2d_backward(Device& dev, const Tensor& grad_out, int H, int W);
Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps);
Tensor softmax(Device& dev, const Tensor& x);
Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target);
Tensor relu_backward(Device& dev, const Tensor& grad_out, const Tensor& x);
Tensor cross_entropy_backward(Device& dev, const Tensor& logits, const Tensor& target);
Tensor transpose(Device& dev, const Tensor& x);
Tensor sum_axis0(Device& dev, const Tensor& x);
void bn_forward_stats(Device& dev, const Tensor& x, Tensor& mean, Tensor& var);
void bn_update_running_stats(Device& dev, Tensor& running_mean, Tensor& running_var,
                             const Tensor& batch_mean, const Tensor& batch_var,
                             float momentum);
void sgd_update(Tensor& param, const Tensor& grad, float lr);
void sgd_momentum_update(Tensor& param, const Tensor& grad, Tensor& velocity,
                         float lr, float momentum);
Tensor conv2d_grad_input(Device& dev, const Tensor& grad_out, const Tensor& x,
                         const Tensor& w, int stride, int padding);
Tensor conv2d_grad_weight(Device& dev, const Tensor& grad_out, const Tensor& x,
                          const Tensor& w, int stride, int padding);
Tensor conv2d_grad_bias(Device& dev, const Tensor& grad_out);
Tensor col2im(Device& dev, const Tensor& col, int N, int Cin, int H, int W,
              int KH, int KW, int stride, int padding);
Tensor grad_out_reshape(Device& dev, const Tensor& grad_out);
Tensor im2col(Device& dev, const Tensor& x, int KH, int KW, int stride, int padding);
Tensor batchnorm_backward(Device& dev, const Tensor& x, const Tensor& grad_out,
                          const Tensor& gamma, const Tensor& mean, const Tensor& var,
                          float eps, Tensor& grad_gamma, Tensor& grad_beta);

}  // namespace opendll::cpu_ops
