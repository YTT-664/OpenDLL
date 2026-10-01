#include "opendll/tensor.hpp"

#include <stdexcept>
#include <utility>

namespace opendll {

Tensor::Tensor(Device& device, std::vector<int64_t> shape)
    : device_(&device), shape_(std::move(shape)) {
    if (numel() == 0) {
        throw std::invalid_argument("Tensor shape has zero elements");
    }
    buffer_ = device.alloc(numel() * sizeof(float));
}

std::size_t Tensor::numel() const {
    std::size_t n = 1;
    for (auto d : shape_) {
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

void Tensor::upload(const float* data) {
    device_->main_queue().upload(*buffer_, data, numel() * sizeof(float));
}

void Tensor::upload(const std::vector<float>& data) {
    if (data.size() != numel()) {
        throw std::invalid_argument("Tensor::upload size mismatch");
    }
    upload(data.data());
}

void Tensor::download(std::vector<float>& out) const {
    out.resize(numel());
    download(out.data());
}

void Tensor::download(float* data) const {
    device_->main_queue().download(*buffer_, data, numel() * sizeof(float));
}

Tensor Tensor::view(std::vector<int64_t> shape) const {
    std::size_t n = 1;
    for (auto d : shape) {
        n *= static_cast<std::size_t>(d);
    }
    if (n != numel()) {
        throw std::invalid_argument("Tensor::view size mismatch");
    }
    Tensor t;
    t.device_ = device_;
    t.shape_ = std::move(shape);
    t.buffer_ = buffer_;
    return t;
}

}  // namespace opendll
