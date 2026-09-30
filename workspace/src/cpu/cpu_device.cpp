#include "cpu_device.hpp"

#include <cstring>
#include <stdexcept>

namespace opendll {

DeviceInfo CpuDevice::info() const {
    DeviceInfo info;
    info.backend = Backend::CPU;
    info.name = "CPU reference backend";
    return info;
}

std::unique_ptr<Buffer> CpuDevice::alloc(std::size_t bytes) {
    return std::make_unique<CpuBuffer>(bytes);
}

std::unique_ptr<Kernel> CpuDevice::compile(const std::string& source) {
    // M0：CPU 后端不执行 GLSL，仅保存源码占位。
    // 真实算子的 CPU 参考执行将在 M1 引入（用于数值 diff 验证）。
    return std::make_unique<CpuKernel>(source);
}

void CpuDevice::CpuQueue::upload(Buffer& dst, const void* src, std::size_t bytes,
                                 std::size_t offset) {
    auto& buf = static_cast<CpuBuffer&>(dst);
    if (offset + bytes > buf.size()) {
        throw std::out_of_range("CpuQueue::upload out of range");
    }
    std::memcpy(buf.data() + offset, src, bytes);
}

void CpuDevice::CpuQueue::download(const Buffer& src, void* dst, std::size_t bytes,
                                   std::size_t offset) {
    const auto& buf = static_cast<const CpuBuffer&>(src);
    if (offset + bytes > buf.size()) {
        throw std::out_of_range("CpuQueue::download out of range");
    }
    std::memcpy(dst, buf.data() + offset, bytes);
}

void CpuDevice::CpuQueue::dispatch(const Kernel&, std::uint32_t, std::uint32_t,
                                   std::uint32_t,
                                   const std::vector<BufferBinding>&) {
    // M0 占位：CPU 后端的真实算子调度在 M1 实现。
}

}  // namespace opendll
