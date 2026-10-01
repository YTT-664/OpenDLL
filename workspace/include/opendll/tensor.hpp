#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "opendll/device.hpp"

namespace opendll {

// M1：contiguous float32 tensor。轻量句柄 + 共享设备存储（拷贝共享底层 buffer）。
class Tensor {
public:
    Tensor() = default;
    Tensor(Device& device, std::vector<int64_t> shape);

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int i) const { return shape_.at(i); }
    std::size_t ndim() const { return shape_.size(); }
    std::size_t numel() const;
    bool valid() const { return buffer_ != nullptr; }

    Device& device() const { return *device_; }
    Buffer& buffer() const { return *buffer_; }

    void upload(const float* data);
    void upload(const std::vector<float>& data);
    void download(std::vector<float>& out) const;
    void download(float* data) const;

    // 返回共享底层 buffer 的 view（改 shape，numel 不变）。
    Tensor view(std::vector<int64_t> shape) const;

private:
    Device* device_ = nullptr;
    std::vector<int64_t> shape_;
    std::shared_ptr<Buffer> buffer_;
};

}  // namespace opendll
