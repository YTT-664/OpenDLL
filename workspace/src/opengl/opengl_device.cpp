#include "opengl_device.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace opendll {

OpenGLBuffer::OpenGLBuffer(GLuint handle, std::size_t size, std::shared_ptr<GLState> state)
    : handle_(handle), size_(size), state_(std::move(state)) {}

OpenGLBuffer::~OpenGLBuffer() {
    if (state_ && handle_ != 0) {
        state_->wgl.makeCurrent();
        state_->funcs.deleteBuffers(1, &handle_);
    }
}

OpenGLKernel::OpenGLKernel(GLuint program, std::shared_ptr<GLState> state)
    : program_(program), state_(std::move(state)) {}

OpenGLKernel::~OpenGLKernel() {
    if (state_ && program_ != 0) {
        state_->funcs.deleteProgram(program_);
    }
}

OpenGLDevice::~OpenGLDevice() = default;

std::unique_ptr<Device> OpenGLDevice::create() {
    auto dev = std::unique_ptr<OpenGLDevice>(new OpenGLDevice());
    if (!dev->init()) {
        return nullptr;
    }
    return dev;
}

bool OpenGLDevice::init() {
    state_ = std::make_shared<GLState>();
    if (!state_->wgl.create(4, 3)) {
        return false;
    }
    if (!state_->wgl.makeCurrent()) {
        return false;
    }
    if (!gl::load(state_->funcs)) {
        return false;
    }
    queue_.bind(&state_->funcs, &state_->wgl);

    GLint v = 0;
    state_->funcs.getIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &v);
    info_.max_workgroup_size = static_cast<std::uint32_t>(v);
    v = 0;
    state_->funcs.getIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &v);
    info_.max_shared_memory = static_cast<std::uint32_t>(v);

    const char* version = reinterpret_cast<const char*>(state_->funcs.getString(GL_VERSION));
    const char* renderer = reinterpret_cast<const char*>(state_->funcs.getString(GL_RENDERER));
    info_.backend = Backend::OpenGL;
    info_.name = std::string("OpenGL ") + (version ? version : "?") + " / " +
                 (renderer ? renderer : "unknown renderer");
    return true;
}

DeviceInfo OpenGLDevice::info() const {
    return info_;
}

std::unique_ptr<Buffer> OpenGLDevice::alloc(std::size_t bytes) {
    state_->wgl.makeCurrent();
    GLuint handle = 0;
    state_->funcs.genBuffers(1, &handle);
    state_->funcs.bindBuffer(GL_SHADER_STORAGE_BUFFER, handle);
    state_->funcs.bufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr,
                             GL_DYNAMIC_COPY);
    return std::make_unique<OpenGLBuffer>(handle, bytes, state_);
}

std::unique_ptr<Kernel> OpenGLDevice::compile(const std::string& source) {
    state_->wgl.makeCurrent();

    GLuint shader = state_->funcs.createShader(GL_COMPUTE_SHADER);
    const char* src = source.c_str();
    state_->funcs.shaderSource(shader, 1, &src, nullptr);
    state_->funcs.compileShader(shader);

    GLint ok = 0;
    state_->funcs.getShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        state_->funcs.getShaderInfoLog(shader, sizeof(log), nullptr, log);
        state_->funcs.deleteShader(shader);
        throw std::runtime_error(std::string("OpenGL shader compile failed: ") + log);
    }

    GLuint program = state_->funcs.createProgram();
    state_->funcs.attachShader(program, shader);
    state_->funcs.linkProgram(program);
    state_->funcs.deleteShader(shader);  // 链接后即可释放

    ok = 0;
    state_->funcs.getProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        state_->funcs.getProgramInfoLog(program, sizeof(log), nullptr, log);
        state_->funcs.deleteProgram(program);
        throw std::runtime_error(std::string("OpenGL program link failed: ") + log);
    }

    return std::make_unique<OpenGLKernel>(program, state_);
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
