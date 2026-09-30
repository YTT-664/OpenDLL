#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <windows.h>

#include "opendll/device.hpp"
#include "gl_loader.hpp"
#include "wgl_context.hpp"

namespace opendll {

// GL 资源（buffer/program）在设备析构、上下文销毁时由驱动统一回收，
// 此处不主动释放，避免资源生命周期先于设备导致的悬空调用。
// M2 引入内存池后会改为显式管理。
class OpenGLBuffer final : public Buffer {
public:
    OpenGLBuffer(GLuint handle, std::size_t size) : handle_(handle), size_(size) {}
    std::size_t size() const override { return size_; }
    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
    std::size_t size_ = 0;
};

class OpenGLKernel final : public Kernel {
public:
    explicit OpenGLKernel(GLuint program) : program_(program) {}
    GLuint program() const { return program_; }

private:
    GLuint program_ = 0;
};

class OpenGLDevice final : public Device {
public:
    ~OpenGLDevice() override;

    DeviceInfo info() const override;
    std::unique_ptr<Buffer> alloc(std::size_t bytes) override;
    std::unique_ptr<Kernel> compile(const std::string& source) override;
    CommandQueue& main_queue() override { return queue_; }
    void synchronize() override { queue_.synchronize(); }

    // 创建并初始化设备，失败返回 nullptr。
    static std::unique_ptr<Device> create();

private:
    bool init();

    gl::WGLContext context_;
    gl::Funcs gl_;
    DeviceInfo info_;

    class GLQueue final : public CommandQueue {
    public:
        GLQueue(gl::Funcs* f, gl::WGLContext* ctx) : f_(f), ctx_(ctx) {}

        void upload(Buffer& dst, const void* src, std::size_t bytes,
                    std::size_t offset = 0) override;
        void download(const Buffer& src, void* dst, std::size_t bytes,
                      std::size_t offset = 0) override;
        void dispatch(const Kernel& k, std::uint32_t gx, std::uint32_t gy,
                      std::uint32_t gz,
                      const std::vector<BufferBinding>& bindings) override;
        void synchronize() override;

    private:
        gl::Funcs* f_;
        gl::WGLContext* ctx_;
    };

    GLQueue queue_{&gl_, &context_};
};

}  // namespace opendll
