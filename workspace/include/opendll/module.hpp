#pragma once

#include <utility>
#include <vector>

#include "opendll/tensor.hpp"

namespace opendll {

// Module 基类：forward 组合算子，backward 传播梯度（静态计算图、手动反向）。
class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0;
    // 返回 (参数, 梯度) 对，供优化器更新。
    virtual std::vector<std::pair<Tensor*, Tensor*>> parameters() = 0;
};

// 全连接层：y = x @ w^T + b。w: [out, in]，b: [out]。
class Linear final : public Module {
public:
    Linear(Device& dev, int in_features, int out_features);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }

private:
    Device& dev_;
    Tensor weight_;  // [out, in]
    Tensor bias_;    // [out]
    Tensor grad_w_;  // [out, in]
    Tensor grad_b_;  // [out]
    Tensor input_;   // 缓存 forward 输入
};

// ReLU 激活（无参数）。
class ReLU final : public Module {
public:
    explicit ReLU(Device& dev) : dev_(dev) {}

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override { return {}; }

private:
    Device& dev_;
    Tensor input_;
};

}  // namespace opendll
