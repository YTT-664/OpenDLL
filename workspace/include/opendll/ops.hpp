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

// 2D 卷积（含 bias）。x: [N, C_in, H, W]，w: [C_out, C_in, KH, KW]，b: [C_out]
// 输出 [N, C_out, H_out, W_out]。stride 为步长，padding 为每边填充。
Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding);

// 全连接层（含 bias）。x: [M, K]，w: [N, K]，b: [N] -> y: [M, N]，y = x @ w^T + b
Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b);

}  // namespace opendll
