// 测试 OpenGL compute 管线能否调用 NVIDIA 独显（WGL_NV_gpu_affinity）。
#include <windows.h>
#include <GL/glcorearb.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// ---- WGL_NV_gpu_affinity 手动定义（不依赖 wglext.h）----
DECLARE_HANDLE(HGPUNV);

typedef struct _GPU_DEVICE {
    DWORD cb;
    CHAR DeviceName[32];
    CHAR DeviceString[128];
    DWORD Flags;
    RECT rcVirtualScreen;
} GPU_DEVICE;

typedef BOOL(WINAPI* PFNWGLENUMGPUSNVPROC)(UINT iGpuIndex, HGPUNV* phGpu);
typedef BOOL(WINAPI* PFNWGLENUMGPUDEVICESNVPROC)(HGPUNV hGpu, UINT iDeviceIndex,
                                                GPU_DEVICE* lpGpuDevice);
typedef HDC(WINAPI* PFNWGLCREATEAFFINITYDCNVPROC)(const HGPUNV* phGpuList);
typedef BOOL(WINAPI* PFNWGLDELETEDCNVPROC)(HDC hdc);
typedef HGLRC(WINAPI* PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC hDC, HGLRC hShareContext,
                                                         const int* attribList);

constexpr int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
constexpr int WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
constexpr int WGL_CONTEXT_PROFILE_MASK_ARB = 0x9126;
constexpr int WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x00000001;

namespace {

const char* kWindowClass = "OpenDLL-GPUTest-Window";

struct Context {
    HWND hwnd = nullptr;
    HDC hdc = nullptr;
    HGLRC hglrc = nullptr;

    void destroy() {
        if (hglrc) {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(hglrc);
            hglrc = nullptr;
        }
        if (hdc && hwnd) {
            ReleaseDC(hwnd, hdc);
            hdc = nullptr;
        }
        if (hwnd) {
            DestroyWindow(hwnd);
            hwnd = nullptr;
        }
    }
};

// 创建一个 4.3 核心上下文（在给定 DC 上）。
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
    return temp;  // 回退
}

// 在指定 DC 上跑一个 compute shader（buffer 每个元素 +1），验证管线可用。
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
    PFNGLMAPBUFFERRANGEPROC glMapBufferRange =
        (PFNGLMAPBUFFERRANGEPROC)wglGetProcAddress("glMapBufferRange");
    PFNGLUNMAPBUFFERPROC glUnmapBuffer =
        (PFNGLUNMAPBUFFERPROC)wglGetProcAddress("glUnmapBuffer");
    PFNGLMEMORYBARRIERPROC glMemoryBarrier =
        (PFNGLMEMORYBARRIERPROC)wglGetProcAddress("glMemoryBarrier");

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

    void* mapped =
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(data), GL_MAP_READ_BIT);
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
    // 1. 注册窗口类 + 创建隐藏窗口
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

    // 2. 临时上下文（加载扩展）
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pf, &pfd);
    HGLRC temp = wglCreateContext(hdc);
    wglMakeCurrent(hdc, temp);

    PFNGLGETSTRINGPROC p_glGetString = (PFNGLGETSTRINGPROC)wglGetProcAddress("glGetString");
    if (!p_glGetString) {
        p_glGetString = (PFNGLGETSTRINGPROC)GetProcAddress(GetModuleHandleA("opengl32.dll"),
                                                           "glGetString");
    }

    // 3. 加载 WGL_NV_gpu_affinity。临时上下文默认走集显，wglGetProcAddress 与 opengl32.dll
    //    均不提供；需从 NVIDIA ICD（nvoglv64.dll）直接加载。
    HMODULE opengl32 = GetModuleHandleA("opengl32.dll");
    HMODULE nvogl = LoadLibraryA("nvoglv64.dll");
    if (!nvogl) {
        const std::filesystem::path base =
            "C:/Windows/System32/DriverStore/FileRepository";
        std::error_code ec;
        if (std::filesystem::exists(base, ec)) {
            for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
                if (!entry.is_directory()) {
                    continue;
                }
                const auto candidate = entry.path() / "nvoglv64.dll";
                if (std::filesystem::exists(candidate)) {
                    nvogl = LoadLibraryA(candidate.string().c_str());
                    if (nvogl) {
                        break;
                    }
                }
            }
        }
    }
    if (nvogl) {
        std::printf("[GPU] loaded NVIDIA ICD (nvoglv64.dll)\n");
    }
    auto load_wgl = [opengl32, nvogl](const char* name) -> void* {
        void* p = reinterpret_cast<void*>(wglGetProcAddress(name));
        if (!p && opengl32) {
            p = reinterpret_cast<void*>(GetProcAddress(opengl32, name));
        }
        if (!p && nvogl) {
            p = reinterpret_cast<void*>(GetProcAddress(nvogl, name));
        }
        return p;
    };
    auto enumGpus = reinterpret_cast<PFNWGLENUMGPUSNVPROC>(load_wgl("wglEnumGpusNV"));
    auto enumGpuDevices =
        reinterpret_cast<PFNWGLENUMGPUDEVICESNVPROC>(load_wgl("wglEnumGpuDevicesNV"));
    auto createAffinityDC =
        reinterpret_cast<PFNWGLCREATEAFFINITYDCNVPROC>(load_wgl("wglCreateAffinityDCNV"));
    auto deleteDC = reinterpret_cast<PFNWGLDELETEDCNVPROC>(load_wgl("wglDeleteDCNV"));

    if (!enumGpus || !enumGpuDevices || !createAffinityDC) {
        std::printf("[GPU] WGL_NV_gpu_affinity NOT supported\n");
        wglDeleteContext(temp);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        return 2;
    }
    std::printf("[GPU] WGL_NV_gpu_affinity supported\n");

    // 4. 枚举所有 GPU
    std::vector<HGPUNV> gpus;
    for (UINT i = 0; i < 8; ++i) {
        HGPUNV g = nullptr;
        if (!enumGpus(i, &g)) {
            break;
        }
        gpus.push_back(g);
        GPU_DEVICE dev = {};
        dev.cb = sizeof(dev);
        if (enumGpuDevices(g, 0, &dev)) {
            std::printf("[GPU] index %u: %s (%s)\n", i, dev.DeviceString, dev.DeviceName);
        }
    }
    std::printf("[GPU] total GPUs enumerated: %zu\n", gpus.size());

    // 5. 找到 NVIDIA GPU
    int nvidia_idx = -1;
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        GPU_DEVICE dev = {};
        dev.cb = sizeof(dev);
        if (enumGpuDevices(gpus[i], 0, &dev) && strstr(dev.DeviceString, "NVIDIA") != nullptr) {
            nvidia_idx = static_cast<int>(i);
            break;
        }
    }

    if (nvidia_idx < 0) {
        std::printf("[GPU] no NVIDIA GPU found\n");
        wglDeleteContext(temp);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        return 3;
    }
    std::printf("[GPU] NVIDIA GPU at index %d\n", nvidia_idx);

    // 6. 在 NVIDIA GPU 上创建 affinity DC + 上下文
    HDC affinityDC = createAffinityDC(&gpus[nvidia_idx]);
    if (!affinityDC) {
        std::printf("[GPU] wglCreateAffinityDCNV failed\n");
        wglDeleteContext(temp);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        return 4;
    }

    HGLRC ctx = create_context_on_dc(affinityDC, 4, 3);
    if (!ctx) {
        std::printf("[GPU] create 4.3 context on NVIDIA failed\n");
        deleteDC(affinityDC);
        wglDeleteContext(temp);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        return 5;
    }

    // 7. 读取 renderer
    const char* renderer =
        reinterpret_cast<const char*>(p_glGetString(GL_RENDERER));
    const char* version = reinterpret_cast<const char*>(p_glGetString(GL_VERSION));
    std::printf("[GPU] renderer: %s\n", renderer ? renderer : "?");
    std::printf("[GPU] version: %s\n", version ? version : "?");

    const bool is_nvidia = renderer && strstr(renderer, "NVIDIA") != nullptr;
    std::printf("[GPU] NVIDIA discrete GPU selected: %s\n", is_nvidia ? "YES" : "NO");

    // 8. 跑 compute shader 验证
    const bool compute_ok = run_compute_check();
    std::printf("[GPU] compute shader dispatch: %s\n", compute_ok ? "PASS" : "FAIL");

    // 清理
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(ctx);
    deleteDC(affinityDC);
    wglDeleteContext(temp);
    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);

    std::printf("[GPU] %s\n", (is_nvidia && compute_ok) ? "RESULT: PASS" : "RESULT: FAIL");
    return (is_nvidia && compute_ok) ? 0 : 1;
}
