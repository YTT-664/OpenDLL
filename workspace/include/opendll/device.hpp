#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace opendll {

// 后端类型。OpenGL 为主计算后端，CPU 为数值参考后端。
enum class Backend { OpenGL, CPU };

struct DeviceInfo {
    Backend backend = Backend::CPU;
    std::string name;
    std::uint32_t max_workgroup_size = 0;  // 单个 workgroup 最大线程数
    std::uint32_t max_shared_memory = 0;   // shared memory 上限（字节）
};

// 设备内存块。后端内部持有真实句柄，抽象层不暴露。
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual std::size_t size() const = 0;
};

// 已编译的 compute kernel（对 OpenGL 后端是 linked program，对 CPU 后端是占位）。
class Kernel {
public:
    virtual ~Kernel() = default;
};

// 同步点。
class Event {
public:
    virtual ~Event() = default;
    virtual void wait() = 0;
};

// kernel 参数绑定：把某个 Buffer 绑定到 shader 的 binding point。
struct BufferBinding {
    std::uint32_t binding = 0;
    Buffer* buffer = nullptr;
};

// 命令队列：所有设备操作的唯一入口。M0 同步执行，接口语义为异步预留。
class CommandQueue {
public:
    virtual ~CommandQueue() = default;

    // 主机 -> 设备拷贝
    virtual void upload(Buffer& dst, const void* src, std::size_t bytes,
                        std::size_t offset = 0) = 0;
    // 设备 -> 主机拷贝
    virtual void download(const Buffer& src, void* dst, std::size_t bytes,
                          std::size_t offset = 0) = 0;
    // 分派 kernel
    virtual void dispatch(const Kernel& k, std::uint32_t gx, std::uint32_t gy,
                          std::uint32_t gz,
                          const std::vector<BufferBinding>& bindings) = 0;
    // 阻塞至队列清空
    virtual void synchronize() = 0;
};

// 顶层设备：创建资源与后端对象。
class Device {
public:
    virtual ~Device() = default;

    virtual DeviceInfo info() const = 0;
    virtual std::unique_ptr<Buffer> alloc(std::size_t bytes) = 0;
    virtual std::unique_ptr<Kernel> compile(const std::string& source) = 0;
    virtual CommandQueue& main_queue() = 0;
    virtual void synchronize() = 0;

    // 工厂方法：按后端类型创建设备。失败返回 nullptr。
    static std::unique_ptr<Device> create(Backend b);
};

}  // namespace opendll
