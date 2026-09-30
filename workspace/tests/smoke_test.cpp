#include <cstdint>
#include <iostream>
#include <vector>

#include "opendll/device.hpp"

using namespace opendll;

// trivial compute shader：把 buffer 里每个 float +1
static const char* kAddOneShader = R"(
#version 430 core
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data {
    float values[];
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < values.length()) {
        values[idx] += 1.0;
    }
}
)";

int main() {
    // 1. CPU 后端：内存往返
    {
        auto dev = Device::create(Backend::CPU);
        if (!dev) {
            std::cerr << "[CPU] device creation failed\n";
            return 1;
        }
        auto buf = dev->alloc(4 * sizeof(float));
        std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
        dev->main_queue().upload(*buf, in.data(), in.size() * sizeof(float));
        std::vector<float> out(4);
        dev->main_queue().download(*buf, out.data(), out.size() * sizeof(float));
        bool ok = (in == out);
        std::cout << "[CPU] upload/download roundtrip: " << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) return 1;
    }

    // 2. OpenGL 后端：compile + dispatch + download
    {
        auto dev = Device::create(Backend::OpenGL);
        if (!dev) {
            std::cerr << "[OpenGL] device creation failed (no GL 4.3 context?)\n";
            return 2;
        }
        std::cout << "[OpenGL] " << dev->info().name << "\n";
        std::cout << "[OpenGL] max workgroup size=" << dev->info().max_workgroup_size
                  << " shared mem=" << dev->info().max_shared_memory << "\n";

        constexpr std::size_t N = 8;
        auto buf = dev->alloc(N * sizeof(float));
        std::vector<float> in(N);
        for (std::size_t i = 0; i < N; ++i) in[i] = static_cast<float>(i);
        dev->main_queue().upload(*buf, in.data(), in.size() * sizeof(float));

        auto kernel = dev->compile(kAddOneShader);
        if (!kernel) {
            std::cerr << "[OpenGL] kernel compile failed\n";
            return 3;
        }

        std::vector<BufferBinding> bindings = {{0u, buf.get()}};
        dev->main_queue().dispatch(*kernel, static_cast<std::uint32_t>(N / 64 + 1), 1, 1,
                                   bindings);
        dev->main_queue().synchronize();

        std::vector<float> out(N);
        dev->main_queue().download(*buf, out.data(), out.size() * sizeof(float));

        bool ok = true;
        for (std::size_t i = 0; i < N; ++i) {
            if (out[i] != in[i] + 1.0f) {
                ok = false;
                break;
            }
        }
        std::cout << "[OpenGL] compute dispatch (+1): " << (ok ? "PASS" : "FAIL") << "\n";
        if (!ok) {
            for (std::size_t i = 0; i < N; ++i) {
                std::cout << "  out[" << i << "]=" << out[i] << " expected=" << in[i] + 1.0f
                          << "\n";
            }
            return 4;
        }
    }

    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
