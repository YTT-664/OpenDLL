#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <windows.h>

#include "opendll/device.hpp"
#include "gl_loader.hpp"
#include "wgl_context.hpp"

namespace opendll {

// OpenGL 状态（函数指针 + 上下文），用 shared_ptr 共享，
// 保证 Buffer/Kernel 析构时上下文仍然存活，可安全释放 GL 资源。
struct GLState {
    gl::Funcs funcs;
    gl::WGLContext wgl;
};

class OpenGLBuffer final : public Buffer {
public:
    OpenGLBuffer(GLuint handle, std::size_t size, std::shared_ptr<GLState> state);
    ~OpenGLBuffer() override;

    std::size_t size() const override { return size_; }
    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
    std::size_t size_ = 0;
    std::shared_ptr<GLState> state_;
};

class OpenGLKernel final : public Kernel {
public:
    OpenGLKernel(GLuint program, std::shared_ptr<GLState> state);
    ~OpenGLKernel() override;

    GLuint program() const { return program_; }

private:
    GLuint program_ = 0;
    std::shared_ptr<GLState> state_;
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

    std::shared_ptr<GLState> state_;
    DeviceInfo info_;

    class GLQueue final : public CommandQueue {
    public:
        GLQueue() = default;
        void bind(gl::Funcs* f, gl::WGLContext* ctx) {
            f_ = f;
            ctx_ = ctx;
        }

        void upload(Buffer& dst, const void* src, std::size_t bytes,
                    std::size_t offset = 0) override;
        void download(const Buffer& src, void* dst, std::size_t bytes,
                      std::size_t offset = 0) override;
        void dispatch(const Kernel& k, std::uint32_t gx, std::uint32_t gy,
                      std::uint32_t gz,
                      const std::vector<BufferBinding>& bindings) override;
        void synchronize() override;

    private:
        gl::Funcs* f_ = nullptr;
        gl::WGLContext* ctx_ = nullptr;
    };

    GLQueue queue_;
};

}  // namespace opendll
