#include "opendll/module.hpp"

#include <cmath>
#include <random>
#include <utility>

#include "opendll/ops.hpp"

namespace opendll {

Linear::Linear(Device& dev, int in_features, int out_features)
    : dev_(dev),
      weight_(dev, {out_features, in_features}),
      bias_(dev, {out_features}),
      grad_w_(dev, {out_features, in_features}),
      grad_b_(dev, {out_features}) {
    // Xavier uniform 初始化
    const float bound = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-bound, bound);

    std::vector<float> w(static_cast<std::size_t>(out_features) * in_features);
    for (float& v : w) {
        v = dist(rng);
    }
    weight_.upload(w);

    std::vector<float> b(out_features, 0.0f);
    bias_.upload(b);

    std::vector<float> gw(static_cast<std::size_t>(out_features) * in_features, 0.0f);
    grad_w_.upload(gw);
    std::vector<float> gb(out_features, 0.0f);
    grad_b_.upload(gb);
}

Tensor Linear::forward(const Tensor& x) {
    input_ = x;
    return linear(dev_, x, weight_, bias_);
}

Tensor Linear::backward(const Tensor& grad_out) {
    // grad_input = grad_out @ weight_（[M,N] @ [N,K] -> [M,K]）
    Tensor grad_input = matmul(dev_, grad_out, weight_);
    // grad_w = grad_out^T @ input_
    Tensor go_t = transpose(dev_, grad_out);
    Tensor gw = matmul(dev_, go_t, input_);
    // grad_b = sum over batch
    Tensor gb = sum_axis0(dev_, grad_out);

    grad_w_ = std::move(gw);
    grad_b_ = std::move(gb);
    return grad_input;
}

std::vector<std::pair<Tensor*, Tensor*>> Linear::parameters() {
    return {{&weight_, &grad_w_}, {&bias_, &grad_b_}};
}

Tensor ReLU::forward(const Tensor& x) {
    input_ = x;
    return relu(dev_, x);
}

Tensor ReLU::backward(const Tensor& grad_out) {
    return relu_backward(dev_, grad_out, input_);
}

}  // namespace opendll
