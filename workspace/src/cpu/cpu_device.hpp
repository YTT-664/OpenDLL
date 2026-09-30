#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "opendll/device.hpp"

namespace opendll {

class CpuBuffer final : public Buffer {
public:
    explicit CpuBuffer(std::size_t bytes) : data_(bytes) {}
    std::size_t size() const override { return data_.size(); }

    std::byte* data() { return data_.data(); }
    const std::byte* data() const { return data_.data(); }

private:
    std::vector<std::byte> data_;
};

class CpuKernel final : public Kernel {
public:
    explicit CpuKernel(std::string source) : source_(std::move(source)) {}
    const std::string& source() const { return source_; }

private:
    std::string source_;
};

class CpuDevice final : public Device {
public:
    DeviceInfo info() const override;
    std::unique_ptr<Buffer> alloc(std::size_t bytes) override;
    std::unique_ptr<Kernel> compile(const std::string& source) override;
    CommandQueue& main_queue() override { return queue_; }
    void synchronize() override { queue_.synchronize(); }

private:
    class CpuQueue final : public CommandQueue {
    public:
        void upload(Buffer& dst, const void* src, std::size_t bytes,
                    std::size_t offset = 0) override;
        void download(const Buffer& src, void* dst, std::size_t bytes,
                      std::size_t offset = 0) override;
        void dispatch(const Kernel& k, std::uint32_t gx, std::uint32_t gy,
                      std::uint32_t gz,
                      const std::vector<BufferBinding>& bindings) override;
        void synchronize() override {}
    };

    CpuQueue queue_;
};

}  // namespace opendll
