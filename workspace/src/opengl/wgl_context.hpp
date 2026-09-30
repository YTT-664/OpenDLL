#pragma once

#include <windows.h>

namespace opendll::gl {

// 管理一个 OpenGL 上下文：隐藏窗口 + 设备上下文 + 渲染上下文。
class WGLContext {
public:
    WGLContext() = default;
    ~WGLContext();

    WGLContext(const WGLContext&) = delete;
    WGLContext& operator=(const WGLContext&) = delete;

    // 创建指定版本的核心上下文（如 4.3）。
    bool create(int major, int minor);
    bool makeCurrent();
    void destroy();

    bool valid() const { return hglrc_ != nullptr; }
    HGLRC handle() const { return hglrc_; }
    HDC dc() const { return hdc_; }

private:
    HWND hwnd_ = nullptr;
    HDC hdc_ = nullptr;
    HGLRC hglrc_ = nullptr;
};

}  // namespace opendll::gl
