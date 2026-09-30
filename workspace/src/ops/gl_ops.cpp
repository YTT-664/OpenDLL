#include "gl_ops.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace opendll::gl_ops {

namespace {

// ---- GLSL compute shader 源码 ----

const char* kMatmulSrc = R"(
#version 430 core
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer B { float b[]; };
layout(std430, binding = 2) buffer C { float c[]; };
layout(std430, binding = 3) buffer Params { uint M; uint N; uint K; };
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= M || col >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}
)";

const char* kReluSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer Y { float y[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < y.length()) {
        y[idx] = max(x[idx], 0.0);
    }
}
)";

const char* kAddSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer B { float b[]; };
layout(std430, binding = 2) buffer C { float c[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < c.length()) {
        c[idx] = a[idx] + b[idx];
    }
}
)";

const char* kMulSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer B { float b[]; };
layout(std430, binding = 2) buffer C { float c[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < c.length()) {
        c[idx] = a[idx] * b[idx];
    }
}
)";

// 每个 device 一份 kernel 缓存（M1 简化，M2 引入算子注册表后重构）。
struct Cache {
    std::unique_ptr<Kernel> matmul;
    std::unique_ptr<Kernel> relu;
    std::unique_ptr<Kernel> add;
    std::unique_ptr<Kernel> mul;
    std::unique_ptr<Buffer> params;  // matmul 的 {M, N, K}
};

Cache& cache_for(Device& dev) {
    static std::unordered_map<Device*, Cache> cache;
    return cache[&dev];
}

Kernel* get_or_compile(Device& dev, std::unique_ptr<Kernel>& slot, const char* src) {
    if (!slot) {
        slot = dev.compile(src);
    }
    return slot.get();
}

}  // namespace

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b) {
    const std::uint32_t M = static_cast<std::uint32_t>(a.dim(0));
    const std::uint32_t K = static_cast<std::uint32_t>(a.dim(1));
    const std::uint32_t N = static_cast<std::uint32_t>(b.dim(1));

    Tensor c(dev, {static_cast<int64_t>(M), static_cast<int64_t>(N)});

    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.matmul, kMatmulSrc);
    if (!cache.params) {
        cache.params = dev.alloc(3 * sizeof(std::uint32_t));
    }

    const std::uint32_t params[3] = {M, N, K};
    dev.main_queue().upload(*cache.params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &a.buffer()},
        {1u, &b.buffer()},
        {2u, &c.buffer()},
        {3u, cache.params.get()},
    };
    const std::uint32_t gx = (N + 7u) / 8u;
    const std::uint32_t gy = (M + 7u) / 8u;
    dev.main_queue().dispatch(*kernel, gx, gy, 1, bindings);
    return c;
}

Tensor relu(Device& dev, const Tensor& x) {
    Tensor y(dev, x.shape());
    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.relu, kReluSrc);

    const std::vector<BufferBinding> bindings = {{0u, &x.buffer()}, {1u, &y.buffer()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((x.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return y;
}

Tensor add(Device& dev, const Tensor& a, const Tensor& b) {
    Tensor c(dev, a.shape());
    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.add, kAddSrc);

    const std::vector<BufferBinding> bindings = {
        {0u, &a.buffer()}, {1u, &b.buffer()}, {2u, &c.buffer()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((a.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return c;
}

Tensor mul(Device& dev, const Tensor& a, const Tensor& b) {
    Tensor c(dev, a.shape());
    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.mul, kMulSrc);

    const std::vector<BufferBinding> bindings = {
        {0u, &a.buffer()}, {1u, &b.buffer()}, {2u, &c.buffer()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((a.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return c;
}

}  // namespace opendll::gl_ops
