#pragma once

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

}  // namespace opendll
