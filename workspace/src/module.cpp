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

Conv2d::Conv2d(Device& dev, int in_channels, int out_channels, int kernel, int stride,
               int padding)
    : dev_(dev),
      weight_(dev, {out_channels, in_channels, kernel, kernel}),
      bias_(dev, {out_channels}),
      grad_w_(dev, {out_channels, in_channels, kernel, kernel}),
      grad_b_(dev, {out_channels}),
      stride_(stride),
      padding_(padding) {
    // Kaiming（He）初始化
    const float fan_in = static_cast<float>(in_channels * kernel * kernel);
    const float std = std::sqrt(2.0f / fan_in);
    static int seed_counter = 0;
    std::mt19937 rng(42 + seed_counter++);
    std::normal_distribution<float> dist(0.0f, std);

    std::vector<float> w(static_cast<std::size_t>(out_channels) * in_channels * kernel * kernel);
    for (float& v : w) {
        v = dist(rng);
    }
    weight_.upload(w);

    std::vector<float> b(out_channels, 0.0f);
    bias_.upload(b);
    std::vector<float> gw(w.size(), 0.0f);
    grad_w_.upload(gw);
    std::vector<float> gb(out_channels, 0.0f);
    grad_b_.upload(gb);
}

Tensor Conv2d::forward(const Tensor& x) {
    input_ = x;
    return conv2d(dev_, x, weight_, bias_, stride_, padding_);
}

Tensor Conv2d::backward(const Tensor& grad_out) {
    Tensor grad_input = conv2d_grad_input(dev_, grad_out, input_, weight_, stride_, padding_);
    Tensor gw = conv2d_grad_weight(dev_, grad_out, input_, weight_, stride_, padding_);
    Tensor gb = conv2d_grad_bias(dev_, grad_out);
    grad_w_ = std::move(gw);
    grad_b_ = std::move(gb);
    return grad_input;
}

std::vector<std::pair<Tensor*, Tensor*>> Conv2d::parameters() {
    return {{&weight_, &grad_w_}, {&bias_, &grad_b_}};
}

BatchNorm2d::BatchNorm2d(Device& dev, int num_features, float eps, float momentum)
    : dev_(dev),
      gamma_(dev, {num_features}),
      beta_(dev, {num_features}),
      grad_gamma_(dev, {num_features}),
      grad_beta_(dev, {num_features}),
      running_mean_(dev, {num_features}),
      running_var_(dev, {num_features}),
      batch_mean_(dev, {num_features}),
      batch_var_(dev, {num_features}),
      eps_(eps),
      momentum_(momentum) {
    std::vector<float> ones(num_features, 1.0f);
    std::vector<float> zeros(num_features, 0.0f);
    gamma_.upload(ones);
    beta_.upload(zeros);
    grad_gamma_.upload(zeros);
    grad_beta_.upload(zeros);
    running_mean_.upload(zeros);
    running_var_.upload(ones);  // running var 初始 1
    batch_mean_.upload(zeros);
    batch_var_.upload(zeros);
}

Tensor BatchNorm2d::forward(const Tensor& x) {
    input_ = x;

    if (training_) {
        bn_forward_stats(dev_, x, batch_mean_, batch_var_);
        // running stats 的 EMA 更新在设备端就地完成，避免逐 batch 下载统计量
        bn_update_running_stats(dev_, running_mean_, running_var_, batch_mean_, batch_var_,
                                momentum_);
        return batchnorm2d(dev_, x, gamma_, beta_, batch_mean_, batch_var_, eps_);
    }
    return batchnorm2d(dev_, x, gamma_, beta_, running_mean_, running_var_, eps_);
}

Tensor BatchNorm2d::backward(const Tensor& grad_out) {
    const Tensor& mean = training_ ? batch_mean_ : running_mean_;
    const Tensor& var = training_ ? batch_var_ : running_var_;
    return batchnorm_backward(dev_, input_, grad_out, gamma_, mean, var, eps_, grad_gamma_,
                              grad_beta_);
}

std::vector<std::pair<Tensor*, Tensor*>> BatchNorm2d::parameters() {
    return {{&gamma_, &grad_gamma_}, {&beta_, &grad_beta_}};
}

Sequential& Sequential::add(std::shared_ptr<Module> m) {
    modules_.push_back(std::move(m));
    return *this;
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (auto& m : modules_) {
        out = m->forward(out);
    }
    return out;
}

Tensor Sequential::backward(const Tensor& grad_out) {
    Tensor grad = grad_out;
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

std::vector<std::pair<Tensor*, Tensor*>> Sequential::parameters() {
    std::vector<std::pair<Tensor*, Tensor*>> params;
    for (auto& m : modules_) {
        auto p = m->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

}  // namespace opendll
