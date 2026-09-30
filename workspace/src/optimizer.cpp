#include "opendll/optimizer.hpp"

#include <cstddef>
#include <vector>

namespace opendll {

void SGD::step(const std::vector<Module*>& modules) {
    for (auto* m : modules) {
        for (auto& [param, grad] : m->parameters()) {
            std::vector<float> p, g;
            param->download(p);
            grad->download(g);
            for (std::size_t i = 0; i < p.size(); ++i) {
                p[i] -= lr_ * g[i];
            }
            param->upload(p);
        }
    }
}

}  // namespace opendll
