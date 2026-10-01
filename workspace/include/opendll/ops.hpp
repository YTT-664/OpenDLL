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

// 最大池化（正方形窗口）。x: [N, C, H, W] -> [N, C, H_out, W_out]。
Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);

// 平均池化（正方形窗口），对窗口内有效（非 padding）元素取平均。
Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding);

// 批归一化（给定均值/方差）。x: [N, C, H, W]，gamma/beta/mean/var: [C]。
// y = (x - mean) / sqrt(var + eps) * gamma + beta
Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps);

// softmax 沿最后一维。x: [N, C] -> [N, C]（数值稳定：max 减法）。
Tensor softmax(Device& dev, const Tensor& x);

// 交叉熵（融合 log_softmax + nll）。logits: [N, C]，target: [N]（float 存整数标签）
// -> loss: [N]（逐样本损失，未平均）。
Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target);

// ---- backward 算子 ----

// relu 反向：grad_in = grad_out * (x > 0)
Tensor relu_backward(Device& dev, const Tensor& grad_out, const Tensor& x);

// 交叉熵反向：grad_logits = softmax(logits) - onehot(target)，返回 [N, C]
Tensor cross_entropy_backward(Device& dev, const Tensor& logits, const Tensor& target);

// 2D 转置：[M, N] -> [N, M]
Tensor transpose(Device& dev, const Tensor& x);

// 沿第 0 维求和：[M, N] -> [N]
Tensor sum_axis0(Device& dev, const Tensor& x);

// 计算 batch mean / var（batchnorm 训练模式）。x: [N, C, H, W] -> mean[C], var[C]（有偏方差）
void bn_forward_stats(Device& dev, const Tensor& x, Tensor& mean, Tensor& var);

// in-place SGD 更新：param -= lr * grad（在设备端执行，避免逐 batch 下载/上传参数）
void sgd_update(Tensor& param, const Tensor& grad, float lr);

// ---- conv2d 反向（naive）----
// grad_x: 对输入的梯度，shape 同 x [N, Cin, H, W]
Tensor conv2d_grad_input(Device& dev, const Tensor& grad_out, const Tensor& x,
                         const Tensor& w, int stride, int padding);
// grad_w: 对权重的梯度，shape 同 w [Cout, Cin, KH, KW]
Tensor conv2d_grad_weight(Device& dev, const Tensor& grad_out, const Tensor& x,
                          const Tensor& w, int stride, int padding);
// grad_b: 对 bias 的梯度，[Cout]
Tensor conv2d_grad_bias(Device& dev, const Tensor& grad_out);

// col2im：把列矩阵 col [Kcol, Ncol] 累加映射回 [N, Cin, H, W]（im2col 的逆）
Tensor col2im(Device& dev, const Tensor& col, int N, int Cin, int H, int W,
              int KH, int KW, int stride, int padding);

// 布局转换：grad_out [N, Cout, Hout, Wout] -> [Ncol, Cout]（Ncol = N*Hout*Wout）
Tensor grad_out_reshape(Device& dev, const Tensor& grad_out);

// im2col：x [N, Cin, H, W] -> col [Kcol, Ncol]，Kcol = Cin*KH*KW，Ncol = N*Hout*Wout
Tensor im2col(Device& dev, const Tensor& x, int KH, int KW, int stride, int padding);

// ---- batchnorm 反向 ----
// 返回 grad_x；同时把 grad_gamma / grad_beta 累加到对应张量。
Tensor batchnorm_backward(Device& dev, const Tensor& x, const Tensor& grad_out,
                          const Tensor& gamma, const Tensor& mean, const Tensor& var,
                          float eps, Tensor& grad_gamma, Tensor& grad_beta);

}  // namespace opendll
