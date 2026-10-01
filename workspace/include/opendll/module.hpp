#pragma once

#include <memory>
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

// 2D 卷积（含 bias）。weight: [out, in, k, k]，bias: [out]。
class Conv2d final : public Module {
public:
    Conv2d(Device& dev, int in_channels, int out_channels, int kernel, int stride = 1,
           int padding = 0);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;

private:
    Device& dev_;
    Tensor weight_;
    Tensor bias_;
    Tensor grad_w_;
    Tensor grad_b_;
    Tensor input_;
    int stride_ = 1;
    int padding_ = 0;
};

// 2D 批归一化（含 gamma/beta 与 running stats）。
class BatchNorm2d final : public Module {
public:
    BatchNorm2d(Device& dev, int num_features, float eps = 1e-5f, float momentum = 0.1f);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;

    void train(bool on) { training_ = on; }

private:
    Device& dev_;
    Tensor gamma_;
    Tensor beta_;
    Tensor grad_gamma_;
    Tensor grad_beta_;
    Tensor running_mean_;
    Tensor running_var_;
    Tensor batch_mean_;  // forward 缓存（训练用 batch 统计，eval 用 running）
    Tensor batch_var_;
    Tensor input_;       // forward 缓存输入
    float eps_ = 1e-5f;
    float momentum_ = 0.1f;
    bool training_ = true;
};

// 顺序容器：按顺序 forward / 逆序 backward。
class Sequential final : public Module {
public:
    Sequential() = default;
    Sequential& add(std::shared_ptr<Module> m);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace opendll
