#include "opendll/optimizer.hpp"

#include <vector>

#include "opendll/ops.hpp"

namespace opendll {

void SGD::step(const std::vector<Module*>& modules) {
    for (auto* m : modules) {
        for (auto& [param, grad] : m->parameters()) {
            // 设备端 in-place 更新，避免逐 batch 下载/上传参数
            sgd_update(*param, *grad, lr_);
        }
    }
}

void SGDM::step(const std::vector<Module*>& modules) {
    for (auto* m : modules) {
        for (auto& [param, grad] : m->parameters()) {
            auto it = velocity_.find(param);
            if (it == velocity_.end()) {
                // 首次遇到该参数：分配同 shape 的零 velocity
                Tensor v(param->device(), param->shape());
                std::vector<float> zeros(param->numel(), 0.0f);
                v.upload(zeros);
                it = velocity_.emplace(param, std::move(v)).first;
            }
            sgd_momentum_update(*param, *grad, it->second, lr_, momentum_);
        }
    }
}

}  // namespace opendll
