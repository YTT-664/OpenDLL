#pragma once

#include <memory>

#include "opendll/module.hpp"

namespace opendll {

// ResNet18（CIFAR 变体：conv1 为 3x3 stride1、无 maxpool，适配小尺寸输入）。
//
// 构造参数：
//   in_size     输入正方形边长（如 32 表示 32x32）。
//   num_classes 输出类别数；为 0 时不生成分类头，forward 直接输出
//               global avg pool 之后的 [N, 512] 特征向量（骨干网络尾部）。
//
// 手动命名（state_dict key）与 PyTorch 对齐：conv1.weight、bn1.running_mean、
// layer1.0.conv1.weight、layer2.0.downsample.0.weight、fc.weight 等。
class ResNet18 : public Module {
public:
    ResNet18(Device& dev, int in_size, int num_classes);
    ~ResNet18() override;

    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad_out) override;
    std::vector<std::pair<Tensor*, Tensor*>> parameters() override;
    std::vector<StateEntry> collect_state() override;

    // 切换所有 BatchNorm 的 train/eval 模式（eval 用 running stats）。
    void train_mode(bool on);

    int num_classes() const { return num_classes_; }
    int in_size() const { return in_size_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int in_size_;
    int num_classes_;
};

}  // namespace opendll
