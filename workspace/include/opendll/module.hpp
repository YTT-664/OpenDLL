#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opendll/tensor.hpp"

namespace opendll {

// state_dict 的一条记录：命名的张量。is_buffer 标记非训练状态（如 bn 的 running stats）。
struct StateEntry {
    std::string name;
    Tensor* tensor = nullptr;
    bool is_buffer = false;
};

// Module 基类：forward 组合算子，backward 传播梯度（静态计算图、手动反向）。
class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad_out) = 0;
    // 返回 (参数, 梯度) 对，供优化器更新。
    virtual std::vector<std::pair<Tensor*, Tensor*>> parameters() = 0;
    // 收集 state_dict 条目（可训练参数 + running stats 等 buffer）。
    // 默认返回空；叶子层（Linear/Conv2d/BatchNorm/Sequential）override 提供真实条目。
    virtual std::vector<StateEntry> collect_state() { return {}; }

    // 手动命名（state_dict 的 key 前缀，叶子层会追加 .weight/.bias 等后缀）。
    void set_name(const std::string& name) { name_ = name; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
};

// 全连接层：y = x @ w^T + b。w: [out, in]，b: [out]。
class Linear final : public Module {
public:
    Linear(Device& dev, int in_features, int out_features);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
    std::vector<StateEntry> collect_state() override;

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
    std::vector<StateEntry> collect_state() override { return {}; }

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
    std::vector<StateEntry> collect_state() override;

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
    std::vector<StateEntry> collect_state() override;

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

// 2D 最大池化（正方形窗口，无参数）。
class MaxPool2d final : public Module {
public:
    MaxPool2d(Device& dev, int kernel, int stride, int padding)
        : dev_(dev), kernel_(kernel), stride_(stride), padding_(padding) {}

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override { return {}; }
    std::vector<StateEntry> collect_state() override { return {}; }

private:
    Device& dev_;
    int kernel_;
    int stride_;
    int padding_;
    Tensor input_;  // 缓存 forward 输入（backward 重算 argmax 用）
};

// 2D 平均池化（正方形窗口，无参数）。
class AvgPool2d final : public Module {
public:
    AvgPool2d(Device& dev, int kernel, int stride, int padding)
        : dev_(dev), kernel_(kernel), stride_(stride), padding_(padding) {}

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override { return {}; }
    std::vector<StateEntry> collect_state() override { return {}; }

private:
    Device& dev_;
    int kernel_;
    int stride_;
    int padding_;
    Tensor input_;  // 缓存 forward 输入（backward 确定输出 shape 用）
};

// 顺序容器：按顺序 forward / 逆序 backward。
class Sequential final : public Module {
public:
    Sequential() = default;
    Sequential& add(std::shared_ptr<Module> m);

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
    std::vector<StateEntry> collect_state() override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace opendll
