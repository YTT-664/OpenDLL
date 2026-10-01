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

}  // namespace opendll
