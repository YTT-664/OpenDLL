// 测试 OpenGL 管线能否调用 NVIDIA 独显。
// 方式 1：导出 NvOptimusEnablement（Optimus 官方机制，告知驱动走高性能 GPU）。
// 方式 2：WGL_NV_gpu_affinity（已确认在当前驱动中被移除，保留诊断输出）。
#include <windows.h>
#include <GL/glcorearb.h>

#include <cstdio>
#include <cstring>
#include <vector>

// ---- 告知 NVIDIA Optimus 驱动：本进程需要高性能 GPU（RTX 4060）----
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}

constexpr int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
constexpr int WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
constexpr int WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126;
constexpr int WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001;

typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext,
                                                         const int* attribList);

namespace {

const char* kWindowClass = "OpenDLL-GPUTest-Window";

// 在给定 DC 上创建 4.3 核心上下文。
HGLRC create_context_on_dc(HDC hdc, int major, int minor) {
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.iLayerType = PFD_MAIN_PLANE;
    const int pf = ChoosePixelFormat(hdc, &pfd);
    if (pf == 0 || !SetPixelFormat(hdc, pf, &pfd)) {
        return nullptr;
    }
    HGLRC temp = wglCreateContext(hdc);
    if (!temp) {
        return nullptr;
    }
    wglMakeCurrent(hdc, temp);
    auto createAttribs = reinterpret_cast<PFNWGLCREATECONTEXTATTRIBSARBPROC>(
        wglGetProcAddress("wglCreateContextAttribsARB"));
    if (createAttribs) {
        const int attribs[] = {WGL_CONTEXT_MAJOR_VERSION_ARB, major,
                               WGL_CONTEXT_MINOR_VERSION_ARB, minor,
                               WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
        HGLRC ctx = createAttribs(hdc, nullptr, attribs);
        if (ctx) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(temp);
            wglMakeCurrent(hdc, ctx);
            return ctx;
        }
    }
    return temp;
}

bool run_compute_check() {
    PFNGLCREATESHADERPROC glCreateShader =
        (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
    PFNGLSHADERSOURCEPROC glShaderSource =
        (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
    PFNGLCOMPILESHADERPROC glCompileShader =
        (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
    PFNGLCREATEPROGRAMPROC glCreateProgram =
        (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
    PFNGLATTACHSHADERPROC glAttachShader =
        (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
    PFNGLLINKPROGRAMPROC glLinkProgram =
        (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
    PFNGLUSEPROGRAMPROC glUseProgram =
        (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
    PFNGLGENBUFFERSPROC glGenBuffers =
        (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    PFNGLBINDBUFFERPROC glBindBuffer =
        (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    PFNGLBUFFERDATAPROC glBufferData =
        (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
    PFNGLBINDBUFFERBASEPROC glBindBufferBase =
        (PFNGLBINDBUFFERBASEPROC)wglGetProcAddress("glBindBufferBase");
    PFNGLDISPATCHCOMPUTEPROC glDispatchCompute =
        (PFNGLDISPATCHCOMPUTEPROC)wglGetProcAddress("glDispatchCompute");
    PFNGLMEMORYBARRIERPROC glMemoryBarrier =
        (PFNGLMEMORYBARRIERPROC)wglGetProcAddress("glMemoryBarrier");
    PFNGLMAPBUFFERRANGEPROC glMapBufferRange =
        (PFNGLMAPBUFFERRANGEPROC)wglGetProcAddress("glMapBufferRange");
    PFNGLUNMAPBUFFERPROC glUnmapBuffer =
        (PFNGLUNMAPBUFFERPROC)wglGetProcAddress("glUnmapBuffer");

    if (!glCreateShader || !glDispatchCompute || !glMapBufferRange || !glMemoryBarrier) {
        return false;
    }

    const char* src = R"(
#version 430 core
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { float values[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < values.length()) values[idx] += 1.0;
}
)";
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader);
    glLinkProgram(prog);
    glUseProgram(prog);

    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(data), data, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(data), GL_MAP_READ_BIT);
    if (!mapped) {
        return false;
    }
    bool ok = true;
    const float* out = static_cast<const float*>(mapped);
    for (int i = 0; i < 8; ++i) {
        if (out[i] != data[i] + 1.0f) {
            ok = false;
        }
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    return ok;
}

}  // namespace

int main() {
    std::printf("[GPU] NvOptimusEnablement exported = %lu\n",
                static_cast<unsigned long>(NvOptimusEnablement));

    WNDCLASSA wc = {};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = kWindowClass;
    RegisterClassA(&wc);

    HWND hwnd = CreateWindowExA(0, kWindowClass, "GPUTest", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64,
                                nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
    if (!hwnd) {
        std::printf("[GPU] create window failed\n");
        return 1;
    }
    HDC hdc = GetDC(hwnd);

    HGLRC ctx = create_context_on_dc(hdc, 4, 3);
    if (!ctx) {
        std::printf("[GPU] create 4.3 context failed\n");
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        return 2;
    }

    PFNGLGETSTRINGPROC p_glGetString = (PFNGLGETSTRINGPROC)wglGetProcAddress("glGetString");
    if (!p_glGetString) {
        p_glGetString = (PFNGLGETSTRINGPROC)GetProcAddress(GetModuleHandleA("opengl32.dll"),
                                                           "glGetString");
    }
    const char* renderer = reinterpret_cast<const char*>(p_glGetString(GL_RENDERER));
    const char* version = reinterpret_cast<const char*>(p_glGetString(GL_VERSION));
    std::printf("[GPU] renderer: %s\n", renderer ? renderer : "?");
    std::printf("[GPU] version: %s\n", version ? version : "?");

    const bool is_nvidia = renderer && strstr(renderer, "NVIDIA") != nullptr;
    std::printf("[GPU] NVIDIA discrete GPU active: %s\n", is_nvidia ? "YES" : "NO");

    const bool compute_ok = run_compute_check();
    std::printf("[GPU] compute shader dispatch: %s\n", compute_ok ? "PASS" : "FAIL");

    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(ctx);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);

    std::printf("[GPU] %s\n", (is_nvidia && compute_ok) ? "RESULT: PASS" : "RESULT: FAIL");
    return (is_nvidia && compute_ok) ? 0 : 1;
}
