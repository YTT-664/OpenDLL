#include "wgl_context.hpp"

namespace opendll::gl {

namespace {

constexpr const char* kWindowClass = "OpenDLL-GL-Window";

// WGL_ARB_create_context 常量（不依赖 wglext.h，取标准值）
constexpr int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
constexpr int WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
constexpr int WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126;
constexpr int WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001;
constexpr int WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB = 0x00000002;

using PFNWGLCREATECONTEXTATTRIBSARBPROC =
    HGLRC(WINAPI*)(HDC hDC, HGLRC hShareContext, const int* attribList);

bool g_classRegistered = false;

bool registerClass() {
    if (g_classRegistered) {
        return true;
    }
    WNDCLASSA wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = kWindowClass;
    wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    if (RegisterClassA(&wc) == 0) {
        return false;
    }
    g_classRegistered = true;
    return true;
}

}  // namespace

WGLContext::~WGLContext() {
    destroy();
}

bool WGLContext::create(int major, int minor) {
    if (!registerClass()) {
        return false;
    }

    hwnd_ = CreateWindowExA(0, kWindowClass, "OpenDLL", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64,
                            nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
    if (hwnd_ == nullptr) {
        return false;
    }

    hdc_ = GetDC(hwnd_);
    if (hdc_ == nullptr) {
        return false;
    }

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pixelFormat = ChoosePixelFormat(hdc_, &pfd);
    if (pixelFormat == 0 || SetPixelFormat(hdc_, pixelFormat, &pfd) == FALSE) {
        return false;
    }

    // 临时 1.1 上下文，用于加载 wglCreateContextAttribsARB
    HGLRC temp = wglCreateContext(hdc_);
    if (temp == nullptr) {
        return false;
    }
    wglMakeCurrent(hdc_, temp);

    auto createAttribs = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(
        wglGetProcAddress("wglCreateContextAttribsARB"));

    if (createAttribs != nullptr) {
        const int attribs[] = {
            WGL_CONTEXT_MAJOR_VERSION_ARB, major,
            WGL_CONTEXT_MINOR_VERSION_ARB, minor,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB, 0,
            0,
        };
        hglrc_ = createAttribs(hdc_, nullptr, attribs);
        if (hglrc_ != nullptr) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(temp);
            wglMakeCurrent(hdc_, hglrc_);
        } else {
            hglrc_ = temp;  // 回退到兼容上下文（可能不含 compute）
        }
    } else {
        hglrc_ = temp;  // 驱动不支持 ARB_create_context
    }

    return hglrc_ != nullptr;
}

bool WGLContext::makeCurrent() {
    if (hdc_ == nullptr || hglrc_ == nullptr) {
        return false;
    }
    return wglMakeCurrent(hdc_, hglrc_) == TRUE;
}

void WGLContext::destroy() {
    if (hglrc_ != nullptr) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(hglrc_);
        hglrc_ = nullptr;
    }
    if (hdc_ != nullptr && hwnd_ != nullptr) {
        ReleaseDC(hwnd_, hdc_);
        hdc_ = nullptr;
    }
    if (hwnd_ != nullptr) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
}

}  // namespace opendll::gl
