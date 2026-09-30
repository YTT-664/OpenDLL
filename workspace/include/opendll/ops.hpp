#pragma once

#include "opendll/tensor.hpp"

namespace opendll {

// M1 算子（仅 forward，out-of-place）。按 device 后端类型自动分发到对应实现。

// C = A @ B，2D 矩阵乘。A: [M, K]，B: [K, N] -> C: [M, N]
Tensor matmul(Device& dev, const Tensor& a, const Tensor& b);

// 逐元素 y = max(x, 0)
Tensor relu(Device& dev, const Tensor& x);

// 逐元素 c = a + b（同 shape）
Tensor add(Device& dev, const Tensor& a, const Tensor& b);

// 逐元素 c = a * b（同 shape）
Tensor mul(Device& dev, const Tensor& a, const Tensor& b);

}  // namespace opendll
