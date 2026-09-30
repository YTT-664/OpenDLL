#include "gl_ops.hpp"

#include <cstddef>
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

const char* kConv2dSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer W { float w[]; };
layout(std430, binding = 2) buffer B { float b[]; };
layout(std430, binding = 3) buffer Y { float y[]; };
layout(std430, binding = 4) buffer Params {
    int N, Cin, Cout, H, Wd, Hout, Wout, KH, KW, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * Cout * Hout * Wout;
    if (int(idx) >= total) return;
    int ow = int(idx) % Wout;
    int oh = (int(idx) / Wout) % Hout;
    int co = (int(idx) / (Wout * Hout)) % Cout;
    int n = int(idx) / (Wout * Hout * Cout);
    float sum = b[co];
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= Wd) continue;
                sum += x[((n * Cin + ci) * H + ih) * Wd + iw]
                     * w[((co * Cin + ci) * KH + kh) * KW + kw];
            }
        }
    }
    y[idx] = sum;
}
)";

const char* kLinearSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer W { float w[]; };
layout(std430, binding = 2) buffer B { float b[]; };
layout(std430, binding = 3) buffer Y { float y[]; };
layout(std430, binding = 4) buffer Params { int M; int K; int N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (int(idx) >= M * N) return;
    int n = int(idx) % N;
    int m = int(idx) / N;
    float sum = b[n];
    for (int k = 0; k < K; ++k) {
        sum += x[m * K + k] * w[n * K + k];
    }
    y[idx] = sum;
}
)";

// im2col：把输入 [N, Cin, H, W] 展开成列矩阵 [Kcol, Ncol]，
// Kcol = Cin*KH*KW，Ncol = N*Hout*Wout。
const char* kIm2colSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Cb { float cmat[]; };
layout(std430, binding = 2) buffer Params {
    int N, Cin, H, Wd, Hout, Wout, KH, KW, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int Kcol = Cin * KH * KW;
    int Ncol = N * Hout * Wout;
    int total = Kcol * Ncol;
    if (int(idx) >= total) return;
    int ncol = int(idx) % Ncol;
    int kcol = int(idx) / Ncol;
    int kw = kcol % KW;
    int kh = (kcol / KW) % KH;
    int ci = kcol / (KW * KH);
    int ow = ncol % Wout;
    int oh = (ncol / Wout) % Hout;
    int n = ncol / (Wout * Hout);
    int ih = oh * stride + kh - pad;
    int iw = ow * stride + kw - pad;
    float v = 0.0;
    if (ih >= 0 && ih < H && iw >= 0 && iw < Wd) {
        v = x[((n * Cin + ci) * H + ih) * Wd + iw];
    }
    cmat[idx] = v;
}
)";

// conv2d 的 tiled GEMM（16x16）：Y = W @ col，写回 NCHW 布局（含 bias）。
const char* kConvGemmSrc = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Wb { float w[]; };    // [Cout, Kcol]
layout(std430, binding = 1) buffer Cb { float cmat[]; }; // [Kcol, Ncol]
layout(std430, binding = 2) buffer Bb { float b[]; };    // [Cout]
layout(std430, binding = 3) buffer Yb { float y[]; };    // [N, Cout, Hout, Wout]
layout(std430, binding = 4) buffer Params {
    int Cout, Ncol, Kcol, N, Hout, Wout;
};
shared float Ws[16][16];
shared float Cs[16][16];
void main() {
    uint g_row = gl_GlobalInvocationID.y;  // co
    uint g_col = gl_GlobalInvocationID.x;  // ncol
    uint l_row = gl_LocalInvocationID.y;
    uint l_col = gl_LocalInvocationID.x;
    float sum = 0.0;
    for (int t = 0; t < Kcol; t += 16) {
        if (g_row < uint(Cout) && (t + int(l_col)) < Kcol) {
            Ws[l_row][l_col] = w[g_row * Kcol + t + l_col];
        } else {
            Ws[l_row][l_col] = 0.0;
        }
        if ((t + int(l_row)) < Kcol && g_col < uint(Ncol)) {
            Cs[l_row][l_col] = cmat[(t + l_row) * Ncol + g_col];
        } else {
            Cs[l_row][l_col] = 0.0;
        }
        barrier();
        for (int k = 0; k < 16; ++k) {
            sum += Ws[l_row][k] * Cs[k][l_col];
        }
        barrier();
    }
    if (g_row < uint(Cout) && g_col < uint(Ncol)) {
        int ow = int(g_col) % Wout;
        int oh = (int(g_col) / Wout) % Hout;
        int n = int(g_col) / (Wout * Hout);
        y[((n * Cout + int(g_row)) * Hout + oh) * Wout + ow] = sum + b[g_row];
    }
}
)";

// linear 的 tiled GEMM（16x16）：y = x @ w^T + b。w 为 [N, K]（转置访问）。
const char* kLinearGemmSrc = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Xb { float x[]; };  // [M, K]
layout(std430, binding = 1) buffer Wb { float w[]; };  // [N, K]
layout(std430, binding = 2) buffer Bb { float b[]; };  // [N]
layout(std430, binding = 3) buffer Yb { float y[]; };  // [M, N]
layout(std430, binding = 4) buffer Params { int M; int K; int N; };
shared float Xs[16][16];
shared float Ws[16][16];
void main() {
    uint g_row = gl_GlobalInvocationID.y;  // m
    uint g_col = gl_GlobalInvocationID.x;  // n
    uint l_row = gl_LocalInvocationID.y;
    uint l_col = gl_LocalInvocationID.x;
    float sum = 0.0;
    for (int t = 0; t < K; t += 16) {
        if (g_row < uint(M) && (t + int(l_col)) < K) {
            Xs[l_row][l_col] = x[g_row * K + t + l_col];
        } else {
            Xs[l_row][l_col] = 0.0;
        }
        if ((t + int(l_row)) < K && g_col < uint(N)) {
            Ws[l_row][l_col] = w[g_col * K + t + l_row];
        } else {
            Ws[l_row][l_col] = 0.0;
        }
        barrier();
        for (int k = 0; k < 16; ++k) {
            sum += Xs[l_row][k] * Ws[k][l_col];
        }
        barrier();
    }
    if (g_row < uint(M) && g_col < uint(N)) {
        y[g_row * N + g_col] = sum + b[g_col];
    }
}
)";

// 每个 device 一份 kernel 缓存（M1 简化，M2 引入算子注册表后重构）。
struct Cache {
    std::unique_ptr<Kernel> matmul;
    std::unique_ptr<Kernel> relu;
    std::unique_ptr<Kernel> add;
    std::unique_ptr<Kernel> mul;
    std::unique_ptr<Kernel> conv2d;      // naive（保留作参考）
    std::unique_ptr<Kernel> linear;      // naive（保留作参考）
    std::unique_ptr<Kernel> im2col;
    std::unique_ptr<Kernel> conv_gemm;    // conv2d 的 tiled GEMM（16x16）
    std::unique_ptr<Kernel> linear_gemm;  // linear 的 tiled GEMM（16x16）
    std::unique_ptr<Buffer> params;        // matmul 的 {M, N, K}
    std::unique_ptr<Buffer> conv_params;   // conv2d naive 的 11 个 int
    std::unique_ptr<Buffer> linear_params; // linear 的 {M, K, N}（linear_gemm 复用）
    std::unique_ptr<Buffer> im2col_params;    // 10 个 int
    std::unique_ptr<Buffer> conv_gemm_params; // 6 个 int
    std::unique_ptr<Buffer> col_buf;          // im2col 中间缓冲
    std::size_t col_buf_capacity = 0;
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

Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t Cin = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t Cout = static_cast<std::int32_t>(w.dim(0));
    const std::int32_t KH = static_cast<std::int32_t>(w.dim(2));
    const std::int32_t KW = static_cast<std::int32_t>(w.dim(3));
    const std::int32_t Hout = (H + 2 * padding - KH) / stride + 1;
    const std::int32_t Wout = (W + 2 * padding - KW) / stride + 1;
    const std::int32_t Kcol = Cin * KH * KW;
    const std::int32_t Ncol = N * Hout * Wout;

    Tensor y(dev, {N, Cout, Hout, Wout});

    auto& cache = cache_for(dev);

    // 1. im2col：X -> col [Kcol, Ncol]
    Kernel* im2col = get_or_compile(dev, cache.im2col, kIm2colSrc);
    if (!cache.im2col_params) {
        cache.im2col_params = dev.alloc(10 * sizeof(std::int32_t));
    }
    const std::size_t col_bytes = static_cast<std::size_t>(Kcol) * Ncol * sizeof(float);
    if (!cache.col_buf || cache.col_buf_capacity < col_bytes) {
        cache.col_buf = dev.alloc(col_bytes);
        cache.col_buf_capacity = col_bytes;
    }
    const std::int32_t im2col_params[10] = {N, Cin, H, W, Hout, Wout,
                                            KH, KW, stride, padding};
    dev.main_queue().upload(*cache.im2col_params, im2col_params, sizeof(im2col_params));
    {
        const std::vector<BufferBinding> bindings = {
            {0u, &x.buffer()},
            {1u, cache.col_buf.get()},
            {2u, cache.im2col_params.get()},
        };
        const std::uint32_t total = static_cast<std::uint32_t>(Kcol) * Ncol;
        const std::uint32_t gx = (total + 255u) / 256u;
        dev.main_queue().dispatch(*im2col, gx, 1, 1, bindings);
    }

    // 2. tiled GEMM：Y = W @ col，写回 NCHW（含 bias）
    Kernel* gemm = get_or_compile(dev, cache.conv_gemm, kConvGemmSrc);
    if (!cache.conv_gemm_params) {
        cache.conv_gemm_params = dev.alloc(6 * sizeof(std::int32_t));
    }
    const std::int32_t gemm_params[6] = {Cout, Ncol, Kcol, N, Hout, Wout};
    dev.main_queue().upload(*cache.conv_gemm_params, gemm_params, sizeof(gemm_params));
    {
        const std::vector<BufferBinding> bindings = {
            {0u, &w.buffer()},
            {1u, cache.col_buf.get()},
            {2u, &b.buffer()},
            {3u, &y.buffer()},
            {4u, cache.conv_gemm_params.get()},
        };
        const std::uint32_t gx = (static_cast<std::uint32_t>(Ncol) + 15u) / 16u;
        const std::uint32_t gy = (static_cast<std::uint32_t>(Cout) + 15u) / 16u;
        dev.main_queue().dispatch(*gemm, gx, gy, 1, bindings);
    }

    return y;
}

Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b) {
    const std::int32_t M = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t K = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t N = static_cast<std::int32_t>(w.dim(0));

    Tensor y(dev, {M, N});

    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.linear_gemm, kLinearGemmSrc);
    if (!cache.linear_params) {
        cache.linear_params = dev.alloc(3 * sizeof(std::int32_t));
    }

    const std::int32_t params[3] = {M, K, N};
    dev.main_queue().upload(*cache.linear_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()},
        {1u, &w.buffer()},
        {2u, &b.buffer()},
        {3u, &y.buffer()},
        {4u, cache.linear_params.get()},
    };
    const std::uint32_t gx = (static_cast<std::uint32_t>(N) + 15u) / 16u;
    const std::uint32_t gy = (static_cast<std::uint32_t>(M) + 15u) / 16u;
    dev.main_queue().dispatch(*kernel, gx, gy, 1, bindings);
    return y;
}

}  // namespace opendll::gl_ops
