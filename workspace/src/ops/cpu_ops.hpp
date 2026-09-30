#pragma once

#include "opendll/tensor.hpp"

namespace opendll::cpu_ops {

// CPU 参考实现：通过抽象接口 download/upload，用 C++ 循环计算。
// 作为数值正确性的黄金基准，供 OpenGL 后端 diff 对比。

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b);
Tensor relu(Device& dev, const Tensor& x);
Tensor add(Device& dev, const Tensor& a, const Tensor& b);
Tensor mul(Device& dev, const Tensor& a, const Tensor& b);

}  // namespace opendll::cpu_ops
