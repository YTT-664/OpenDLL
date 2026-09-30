#pragma once

#include "opendll/tensor.hpp"

namespace opendll::gl_ops {

// OpenGL 后端算子实现：编译 GLSL compute shader 并 dispatch。
// 仅在 device 为 OpenGL 后端时被调用（见 ops.cpp 分发）。

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b);
Tensor relu(Device& dev, const Tensor& x);
Tensor add(Device& dev, const Tensor& a, const Tensor& b);
Tensor mul(Device& dev, const Tensor& a, const Tensor& b);
Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding);
Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b);
Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);
Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);
Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps);
Tensor softmax(Device& dev, const Tensor& x);
Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target);

}  // namespace opendll::gl_ops
