#include "opengl_device.hpp"

#include <cstring>
#include <stdexcept>

namespace opendll {

OpenGLDevice::~OpenGLDevice() = default;

std::unique_ptr<Device> OpenGLDevice::create() {
    auto dev = std::unique_ptr<OpenGLDevice>(new OpenGLDevice());
    if (!dev->init()) {
        return nullptr;
    }
    return dev;
}

bool OpenGLDevice::init() {
    if (!context_.create(4, 3)) {
        return false;
    }
    if (!context_.makeCurrent()) {
        return false;
    }
    if (!gl::load(gl_)) {
        return false;
    }

    GLint v = 0;
    gl_.getIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &v);
    info_.max_workgroup_size = static_cast<std::uint32_t>(v);
    v = 0;
    gl_.getIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &v);
    info_.max_shared_memory = static_cast<std::uint32_t>(v);

    const char* version = reinterpret_cast<const char*>(gl_.getString(GL_VERSION));
    const char* renderer = reinterpret_cast<const char*>(gl_.getString(GL_RENDERER));
    info_.backend = Backend::OpenGL;
    info_.name = std::string("OpenGL ") + (version ? version : "?") + " / " +
                 (renderer ? renderer : "unknown renderer");
    return true;
}

DeviceInfo OpenGLDevice::info() const {
    return info_;
}

std::unique_ptr<Buffer> OpenGLDevice::alloc(std::size_t bytes) {
    context_.makeCurrent();
    GLuint handle = 0;
    gl_.genBuffers(1, &handle);
    gl_.bindBuffer(GL_SHADER_STORAGE_BUFFER, handle);
    gl_.bufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr,
                   GL_DYNAMIC_COPY);
    return std::make_unique<OpenGLBuffer>(handle, bytes);
}

std::unique_ptr<Kernel> OpenGLDevice::compile(const std::string& source) {
    context_.makeCurrent();

    GLuint shader = gl_.createShader(GL_COMPUTE_SHADER);
    const char* src = source.c_str();
    gl_.shaderSource(shader, 1, &src, nullptr);
    gl_.compileShader(shader);

    GLint ok = 0;
    gl_.getShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        gl_.getShaderInfoLog(shader, sizeof(log), nullptr, log);
        gl_.deleteShader(shader);
        throw std::runtime_error(std::string("OpenGL shader compile failed: ") + log);
    }

    GLuint program = gl_.createProgram();
    gl_.attachShader(program, shader);
    gl_.linkProgram(program);
    gl_.deleteShader(shader);  // 链接后即可释放

    ok = 0;
    gl_.getProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        gl_.getProgramInfoLog(program, sizeof(log), nullptr, log);
        gl_.deleteProgram(program);
        throw std::runtime_error(std::string("OpenGL program link failed: ") + log);
    }

    return std::make_unique<OpenGLKernel>(program);
}

void OpenGLDevice::GLQueue::upload(Buffer& dst, const void* src, std::size_t bytes,
                                   std::size_t offset) {
    auto& buf = static_cast<OpenGLBuffer&>(dst);
    if (offset + bytes > buf.size()) {
        throw std::out_of_range("GLQueue::upload out of range");
    }
    ctx_->makeCurrent();
    f_->bindBuffer(GL_SHADER_STORAGE_BUFFER, buf.handle());
    void* mapped = f_->mapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                      static_cast<GLintptr>(offset),
                                      static_cast<GLsizeiptr>(bytes),
                                      GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);
    if (mapped == nullptr) {
        throw std::runtime_error("glMapBufferRange failed in upload");
    }
    std::memcpy(mapped, src, bytes);
    f_->unmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void OpenGLDevice::GLQueue::download(const Buffer& src, void* dst, std::size_t bytes,
                                     std::size_t offset) {
    const auto& buf = static_cast<const OpenGLBuffer&>(src);
    if (offset + bytes > buf.size()) {
        throw std::out_of_range("GLQueue::download out of range");
    }
    ctx_->makeCurrent();
    f_->bindBuffer(GL_SHADER_STORAGE_BUFFER, buf.handle());
    void* mapped = f_->mapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                      static_cast<GLintptr>(offset),
                                      static_cast<GLsizeiptr>(bytes), GL_MAP_READ_BIT);
    if (mapped == nullptr) {
        throw std::runtime_error("glMapBufferRange failed in download");
    }
    std::memcpy(dst, mapped, bytes);
    f_->unmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void OpenGLDevice::GLQueue::dispatch(const Kernel& k, std::uint32_t gx, std::uint32_t gy,
                                     std::uint32_t gz,
                                     const std::vector<BufferBinding>& bindings) {
    const auto& kernel = static_cast<const OpenGLKernel&>(k);
    ctx_->makeCurrent();
    f_->useProgram(kernel.program());
    for (const auto& b : bindings) {
        auto* buf = static_cast<OpenGLBuffer*>(b.buffer);
        f_->bindBufferBase(GL_SHADER_STORAGE_BUFFER, b.binding, buf->handle());
    }
    f_->dispatchCompute(gx, gy, gz);
    f_->memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void OpenGLDevice::GLQueue::synchronize() {
    ctx_->makeCurrent();
    f_->finish();
}

}  // namespace opendll
