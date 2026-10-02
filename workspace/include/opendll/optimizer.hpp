#pragma once

#include <unordered_map>
#include <vector>

#include "opendll/module.hpp"

namespace opendll {

// 朴素 SGD（无 momentum）。param -= lr * grad。
class SGD {
public:
    explicit SGD(float lr) : lr_(lr) {}

    void step(const std::vector<Module*>& modules);

private:
    float lr_;
};

// SGDM（SGD with momentum）。v = momentum*v + grad; param -= lr*v。
// 惰性为每个参数维护一个 velocity 缓冲（同 shape，初始 0）。
class SGDM {
public:
    SGDM(float lr, float momentum = 0.9f) : lr_(lr), momentum_(momentum) {}

    void step(const std::vector<Module*>& modules);
    void set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }

private:
    float lr_;
    float momentum_;
    std::unordered_map<Tensor*, Tensor> velocity_;
};

}  // namespace opendll
