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

// 最大池化（正方形窗口）。
const char* kMaxPoolSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Yb { float y[]; };
layout(std430, binding = 2) buffer Params {
    int N, C, H, Wd, Hout, Wout, K, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * C * Hout * Wout;
    if (int(idx) >= total) return;
    int ow = int(idx) % Wout;
    int oh = (int(idx) / Wout) % Hout;
    int c = (int(idx) / (Wout * Hout)) % C;
    int n = int(idx) / (Wout * Hout * C);
    float m = -1.0e30;
    for (int kh = 0; kh < K; ++kh) {
        int ih = oh * stride + kh - pad;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < K; ++kw) {
            int iw = ow * stride + kw - pad;
            if (iw < 0 || iw >= Wd) continue;
            m = max(m, x[((n * C + c) * H + ih) * Wd + iw]);
        }
    }
    y[idx] = m;
}
)";

// 平均池化（正方形窗口，对有效元素取平均）。
const char* kAvgPoolSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Yb { float y[]; };
layout(std430, binding = 2) buffer Params {
    int N, C, H, Wd, Hout, Wout, K, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * C * Hout * Wout;
    if (int(idx) >= total) return;
    int ow = int(idx) % Wout;
    int oh = (int(idx) / Wout) % Hout;
    int c = (int(idx) / (Wout * Hout)) % C;
    int n = int(idx) / (Wout * Hout * C);
    float sum = 0.0;
    int cnt = 0;
    for (int kh = 0; kh < K; ++kh) {
        int ih = oh * stride + kh - pad;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < K; ++kw) {
            int iw = ow * stride + kw - pad;
            if (iw < 0 || iw >= Wd) continue;
            sum += x[((n * C + c) * H + ih) * Wd + iw];
            cnt += 1;
        }
    }
    y[idx] = sum / float(cnt);
}
)";

// 批归一化（给定均值/方差）。
const char* kBatchnormSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Gb { float g[]; };
layout(std430, binding = 2) buffer Bb { float b[]; };
layout(std430, binding = 3) buffer Mb { float mean[]; };
layout(std430, binding = 4) buffer Vb { float var[]; };
layout(std430, binding = 5) buffer Yb { float y[]; };
layout(std430, binding = 6) buffer Params { int C; int HW; float eps; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int c = (int(idx) / HW) % C;
    float scale = 1.0 / sqrt(var[c] + eps);
    y[idx] = (x[idx] - mean[c]) * scale * g[c] + b[c];
}
)";

// softmax 沿最后一维（每行一个线程）。
const char* kSoftmaxSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Yb { float y[]; };
layout(std430, binding = 2) buffer Params { int N; int C; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (int(row) >= N) return;
    float m = x[row * C];
    for (int j = 1; j < C; ++j) m = max(m, x[row * C + j]);
    float s = 0.0;
    for (int j = 0; j < C; ++j) {
        y[row * C + j] = exp(x[row * C + j] - m);
        s += y[row * C + j];
    }
    for (int j = 0; j < C; ++j) y[row * C + j] /= s;
}
)";

// 交叉熵（log_softmax + nll，每样本一个线程）。
const char* kCrossEntropySrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Lb { float logits[]; };
layout(std430, binding = 1) buffer Tb { float target[]; };
layout(std430, binding = 2) buffer Yb { float loss[]; };
layout(std430, binding = 3) buffer Params { int N; int C; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (int(row) >= N) return;
    float m = logits[row * C];
    for (int j = 1; j < C; ++j) m = max(m, logits[row * C + j]);
    float s = 0.0;
    for (int j = 0; j < C; ++j) s += exp(logits[row * C + j] - m);
    float logsumexp = m + log(s);
    int t = int(target[row]);
    loss[row] = logsumexp - logits[row * C + t];
}
)";

// relu 反向。
const char* kReluBackwardSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer Xb { float x[]; };
layout(std430, binding = 2) buffer Yb { float y[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < y.length()) {
        y[idx] = (x[idx] > 0.0) ? go[idx] : 0.0;
    }
}
)";

// 交叉熵反向：softmax - onehot。
const char* kCrossEntropyBackwardSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Lb { float logits[]; };
layout(std430, binding = 1) buffer Tb { float target[]; };
layout(std430, binding = 2) buffer Gb { float grad[]; };
layout(std430, binding = 3) buffer Params { int N; int C; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (int(row) >= N) return;
    float m = logits[row * C];
    for (int j = 1; j < C; ++j) m = max(m, logits[row * C + j]);
    float s = 0.0;
    for (int j = 0; j < C; ++j) s += exp(logits[row * C + j] - m);
    int t = int(target[row]);
    for (int j = 0; j < C; ++j) {
        float p = exp(logits[row * C + j] - m) / s;
        grad[row * C + j] = p - (j == t ? 1.0 : 0.0);
    }
}
)";

// 2D 转置。
const char* kTransposeSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Yb { float y[]; };
layout(std430, binding = 2) buffer Params { int M; int N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (int(idx) >= M * N) return;
    int i = int(idx) / N;
    int j = int(idx) % N;
    y[j * M + i] = x[idx];
}
)";

// 沿第 0 维求和：[M, N] -> [N]。
const char* kSumAxis0Src = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Yb { float y[]; };
layout(std430, binding = 2) buffer Params { int M; int N; };
void main() {
    uint j = gl_GlobalInvocationID.x;
    if (int(j) >= N) return;
    float s = 0.0;
    for (int i = 0; i < M; ++i) {
        s += x[i * N + j];
    }
    y[j] = s;
}
)";

// in-place SGD 更新：p -= lr * g。
const char* kSgdUpdateSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Pb { float p[]; };
layout(std430, binding = 1) buffer Gb { float g[]; };
layout(std430, binding = 2) buffer Params { float lr; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < p.length()) {
        p[idx] -= lr * g[idx];
    }
}
)";

// batchnorm running stats 的 EMA 更新（就地）：running = (1-momentum)*running + momentum*batch。
const char* kBnUpdateRunningSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Rm { float rm[]; };  // running_mean（就地）
layout(std430, binding = 1) buffer Rv { float rv[]; };  // running_var（就地）
layout(std430, binding = 2) buffer Bm { float bm[]; };  // batch_mean
layout(std430, binding = 3) buffer Bv { float bv[]; };  // batch_var
layout(std430, binding = 4) buffer Params { float momentum; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < rm.length()) {
        rm[idx] = (1.0 - momentum) * rm[idx] + momentum * bm[idx];
        rv[idx] = (1.0 - momentum) * rv[idx] + momentum * bv[idx];
    }
}
)";

// conv2d 反向：对输入梯度。
const char* kConv2dGradInputSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer Wb { float w[]; };
layout(std430, binding = 2) buffer GXb { float gx[]; };
layout(std430, binding = 3) buffer Params {
    int N, Cin, Cout, H, Wd, Hout, Wout, KH, KW, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * Cin * H * Wd;
    if (int(idx) >= total) return;
    int iw = int(idx) % Wd;
    int ih = (int(idx) / Wd) % H;
    int ci = (int(idx) / (Wd * H)) % Cin;
    int n = int(idx) / (Wd * H * Cin);
    float sum = 0.0;
    for (int co = 0; co < Cout; ++co) {
        for (int kh = 0; kh < KH; ++kh) {
            int num_h = ih + pad - kh;
            if (num_h % stride != 0) continue;
            int oh = num_h / stride;
            if (oh < 0 || oh >= Hout) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int num_w = iw + pad - kw;
                if (num_w % stride != 0) continue;
                int ow = num_w / stride;
                if (ow < 0 || ow >= Wout) continue;
                sum += go[((n * Cout + co) * Hout + oh) * Wout + ow]
                     * w[((co * Cin + ci) * KH + kh) * KW + kw];
            }
        }
    }
    gx[idx] = sum;
}
)";

// conv2d 反向：对权重梯度。
const char* kConv2dGradWeightSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer Xb { float x[]; };
layout(std430, binding = 2) buffer GWb { float gw[]; };
layout(std430, binding = 3) buffer Params {
    int N, Cin, Cout, H, Wd, Hout, Wout, KH, KW, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = Cout * Cin * KH * KW;
    if (int(idx) >= total) return;
    int kw = int(idx) % KW;
    int kh = (int(idx) / KW) % KH;
    int ci = (int(idx) / (KW * KH)) % Cin;
    int co = int(idx) / (KW * KH * Cin);
    float sum = 0.0;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            int ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < Wout; ++ow) {
                int iw = ow * stride - pad + kw;
                if (iw < 0 || iw >= Wd) continue;
                sum += go[((n * Cout + co) * Hout + oh) * Wout + ow]
                     * x[((n * Cin + ci) * H + ih) * Wd + iw];
            }
        }
    }
    gw[idx] = sum;
}
)";

// conv2d 反向：对 bias 梯度。
const char* kConv2dGradBiasSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer GBb { float gb[]; };
layout(std430, binding = 2) buffer Params { int N; int Cout; int Hout; int Wout; };
void main() {
    uint co = gl_GlobalInvocationID.x;
    if (int(co) >= Cout) return;
    float sum = 0.0;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                sum += go[((n * Cout + int(co)) * Hout + oh) * Wout + ow];
            }
        }
    }
    gb[co] = sum;
}
)";

// batchnorm 反向第一步：每 channel 计算 sum(dy) 与 sum(dy*x_hat)，累加 grad_gamma/grad_beta。
const char* kBnBackwardStatsSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer Xb { float x[]; };
layout(std430, binding = 2) buffer Mb { float mean[]; };
layout(std430, binding = 3) buffer Vb { float var[]; };
layout(std430, binding = 4) buffer GGb { float gg[]; };
layout(std430, binding = 5) buffer GBb { float gb[]; };
layout(std430, binding = 6) buffer Stb { float stats[]; };
layout(std430, binding = 7) buffer Params { int N; int C; int HW; float eps; };
shared float s_gy[256];
shared float s_gyx[256];
void main() {
    int c = int(gl_WorkGroupID.x);       // 每个 workgroup 负责一个 channel
    int tid = int(gl_LocalInvocationID.x);
    int M = N * HW;
    float inv_std = 1.0 / sqrt(var[c] + eps);

    float local_gy = 0.0;
    float local_gyx = 0.0;
    for (int i = tid; i < M; i += 256) {
        int n = i / HW;
        int hw = i - n * HW;
        int idx = (n * C + c) * HW + hw;
        float xhat = (x[idx] - mean[c]) * inv_std;
        float gy = go[idx];
        local_gy += gy;
        local_gyx += gy * xhat;
    }

    s_gy[tid] = local_gy;
    s_gyx[tid] = local_gyx;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_gy[tid] += s_gy[tid + s];
            s_gyx[tid] += s_gyx[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        gg[c] += s_gyx[0];
        gb[c] += s_gy[0];
        stats[c] = s_gy[0];
        stats[C + c] = s_gyx[0];
    }
}
)";

// batchnorm 反向第二步：由统计量计算 grad_x。
const char* kBnBackwardGradXSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer Xb { float x[]; };
layout(std430, binding = 2) buffer Mb { float mean[]; };
layout(std430, binding = 3) buffer Vb { float var[]; };
layout(std430, binding = 4) buffer Gb { float gamma[]; };
layout(std430, binding = 5) buffer Stb { float stats[]; };
layout(std430, binding = 6) buffer GXb { float gx[]; };
layout(std430, binding = 7) buffer Params { int N; int C; int HW; float eps; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * C * HW;
    if (int(idx) >= total) return;
    int c = (int(idx) / HW) % C;
    float inv_std = 1.0 / sqrt(var[c] + eps);
    float xhat = (x[idx] - mean[c]) * inv_std;
    float gy = go[idx];
    float sum_gy = stats[c];
    float sum_gy_xhat = stats[C + c];
    float M = float(N * HW);
    gx[idx] = inv_std * gamma[c] * (gy - sum_gy / M - xhat * sum_gy_xhat / M);
}
)";

// batchnorm 前向统计：每 channel 计算 batch mean 与 var（有偏方差）。
const char* kBnForwardStatsSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Xb { float x[]; };
layout(std430, binding = 1) buffer Mb { float mean[]; };
layout(std430, binding = 2) buffer Vb { float var[]; };
layout(std430, binding = 3) buffer Params { int N; int C; int HW; };
shared float s_sum[256];
shared float s_sum2[256];
void main() {
    int c = int(gl_WorkGroupID.x);       // 每个 workgroup 负责一个 channel
    int tid = int(gl_LocalInvocationID.x);
    int M = N * HW;

    float local_sum = 0.0;
    float local_sum2 = 0.0;
    for (int i = tid; i < M; i += 256) {
        int n = i / HW;
        int hw = i - n * HW;
        int idx = (n * C + c) * HW + hw;
        float v = x[idx];
        local_sum += v;
        local_sum2 += v * v;
    }

    s_sum[tid] = local_sum;
    s_sum2[tid] = local_sum2;
    barrier();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum2[tid] += s_sum2[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        float mu = s_sum[0] / float(M);
        mean[c] = mu;
        var[c] = s_sum2[0] / float(M) - mu * mu;
    }
}
)";

// col2im：列矩阵 [Kcol, Ncol] 累加回 [N, Cin, H, W]。
const char* kCol2imSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Colb { float col[]; };
layout(std430, binding = 1) buffer GXb { float gx[]; };
layout(std430, binding = 2) buffer Params {
    int N, Cin, H, Wd, Hout, Wout, KH, KW, stride, pad;
};
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = N * Cin * H * Wd;
    if (int(idx) >= total) return;
    int iw = int(idx) % Wd;
    int ih = (int(idx) / Wd) % H;
    int ci = (int(idx) / (Wd * H)) % Cin;
    int n = int(idx) / (Wd * H * Cin);
    int Kcol = Cin * KH * KW;
    int Ncol = N * Hout * Wout;
    float sum = 0.0;
    for (int kh = 0; kh < KH; ++kh) {
        int num_h = ih + pad - kh;
        if (num_h % stride != 0) continue;
        int oh = num_h / stride;
        if (oh < 0 || oh >= Hout) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int num_w = iw + pad - kw;
            if (num_w % stride != 0) continue;
            int ow = num_w / stride;
            if (ow < 0 || ow >= Wout) continue;
            int p = (n * Hout + oh) * Wout + ow;
            int k = (ci * KH + kh) * KW + kw;
            sum += col[k * Ncol + p];
        }
    }
    gx[idx] = sum;
}
)";

// grad_out 布局转换：[N, Cout, Hout, Wout] -> [Ncol, Cout]。
const char* kGradOutReshapeSrc = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer GOb { float go[]; };
layout(std430, binding = 1) buffer GYb { float gy[]; };
layout(std430, binding = 2) buffer Params { int Cout; int Hout; int Wout; int Ncol; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    int total = Ncol * Cout;
    if (int(idx) >= total) return;
    int co = int(idx) % Cout;
    int p = int(idx) / Cout;
    int ow = p % Wout;
    int oh = (p / Wout) % Hout;
    int n = p / (Wout * Hout);
    gy[idx] = go[((n * Cout + co) * Hout + oh) * Wout + ow];
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
    std::unique_ptr<Kernel> maxpool;
    std::unique_ptr<Kernel> avgpool;
    std::unique_ptr<Kernel> batchnorm;
    std::unique_ptr<Kernel> softmax;
    std::unique_ptr<Kernel> cross_entropy;
    std::unique_ptr<Kernel> relu_backward;
    std::unique_ptr<Kernel> cross_entropy_backward;
    std::unique_ptr<Kernel> transpose;
    std::unique_ptr<Kernel> sum_axis0;
    std::unique_ptr<Kernel> sgd_update;
    std::unique_ptr<Kernel> conv2d_grad_input;
    std::unique_ptr<Kernel> conv2d_grad_weight;
    std::unique_ptr<Kernel> conv2d_grad_bias;
    std::unique_ptr<Kernel> bn_backward_stats;
    std::unique_ptr<Kernel> bn_backward_gradx;
    std::unique_ptr<Kernel> bn_fwd_stats;
    std::unique_ptr<Kernel> bn_update_running;
    std::unique_ptr<Kernel> col2im;
    std::unique_ptr<Kernel> grad_out_reshape;
    std::unique_ptr<Buffer> params;        // matmul 的 {M, N, K}
    std::unique_ptr<Buffer> conv_params;   // conv2d naive 的 11 个 int
    std::unique_ptr<Buffer> linear_params; // linear 的 {M, K, N}（linear_gemm 复用）
    std::unique_ptr<Buffer> im2col_params;    // 10 个 int
    std::unique_ptr<Buffer> conv_gemm_params; // 6 个 int
    std::unique_ptr<Buffer> pool_params;   // 9 个 int（maxpool/avgpool 复用）
    std::unique_ptr<Buffer> bn_params;     // 2 int + 1 float
    std::unique_ptr<Buffer> sm_params;     // 2 个 int（softmax/ce 复用）
    std::unique_ptr<Buffer> sgd_params;    // 1 float（lr）
    std::unique_ptr<Buffer> conv_bwd_params;   // 11 个 int（grad_input/grad_weight 复用）
    std::unique_ptr<Buffer> conv_bias_params;  // 4 个 int
    std::unique_ptr<Buffer> bn_fwd_params;     // 3 个 int（N, C, HW）
    std::unique_ptr<Buffer> bn_running_params; // 1 float（momentum）
    std::unique_ptr<Buffer> bn_bwd_params;     // 3 int + 1 float
    std::unique_ptr<Buffer> bn_stats_buf;      // 2*C float（sum_gy / sum_gy_xhat）
    std::unique_ptr<Buffer> col2im_params;     // 11 个 int
    std::unique_ptr<Buffer> grad_reshape_params; // 4 个 int
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

Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t Hout = (H + 2 * padding - kernel) / stride + 1;
    const std::int32_t Wout = (W + 2 * padding - kernel) / stride + 1;

    Tensor y(dev, {N, C, Hout, Wout});

    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.maxpool, kMaxPoolSrc);
    if (!cache.pool_params) {
        cache.pool_params = dev.alloc(9 * sizeof(std::int32_t));
    }
    const std::int32_t params[9] = {N, C, H, W, Hout, Wout, kernel, stride, padding};
    dev.main_queue().upload(*cache.pool_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &y.buffer()}, {2u, cache.pool_params.get()}};
    const std::uint32_t total = static_cast<std::uint32_t>(N) * C * Hout * Wout;
    const std::uint32_t gx = (total + 255u) / 256u;
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return y;
}

Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t Hout = (H + 2 * padding - kernel) / stride + 1;
    const std::int32_t Wout = (W + 2 * padding - kernel) / stride + 1;

    Tensor y(dev, {N, C, Hout, Wout});

    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.avgpool, kAvgPoolSrc);
    if (!cache.pool_params) {
        cache.pool_params = dev.alloc(9 * sizeof(std::int32_t));
    }
    const std::int32_t params[9] = {N, C, H, W, Hout, Wout, kernel, stride, padding};
    dev.main_queue().upload(*cache.pool_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &y.buffer()}, {2u, cache.pool_params.get()}};
    const std::uint32_t total = static_cast<std::uint32_t>(N) * C * Hout * Wout;
    const std::uint32_t gx = (total + 255u) / 256u;
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return y;
}

Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps) {
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t HW = H * W;

    Tensor y(dev, x.shape());

    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.batchnorm, kBatchnormSrc);
    if (!cache.bn_params) {
        cache.bn_params = dev.alloc(12);
    }
    struct BNParams {
        std::int32_t C;
        std::int32_t HW;
        float eps;
    };
    const BNParams p{C, HW, eps};
    dev.main_queue().upload(*cache.bn_params, &p, sizeof(p));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()},
        {1u, &gamma.buffer()},
        {2u, &beta.buffer()},
        {3u, &mean.buffer()},
        {4u, &var.buffer()},
        {5u, &y.buffer()},
        {6u, cache.bn_params.get()},
    };
    const std::uint32_t total = static_cast<std::uint32_t>(x.numel());
    const std::uint32_t gx = (total + 255u) / 256u;
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return y;
}

Tensor softmax(Device& dev, const Tensor& x) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));

    Tensor y(dev, x.shape());

    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.softmax, kSoftmaxSrc);
    if (!cache.sm_params) {
        cache.sm_params = dev.alloc(2 * sizeof(std::int32_t));
    }
    const std::int32_t params[2] = {N, C};
    dev.main_queue().upload(*cache.sm_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &y.buffer()}, {2u, cache.sm_params.get()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 255u) / 256u);
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return y;
}

Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target) {
    const std::int32_t N = static_cast<std::int32_t>(logits.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(logits.dim(1));

    Tensor loss(dev, {N});

    auto& cache = cache_for(dev);
    Kernel* kernel = get_or_compile(dev, cache.cross_entropy, kCrossEntropySrc);
    if (!cache.sm_params) {
        cache.sm_params = dev.alloc(2 * sizeof(std::int32_t));
    }
    const std::int32_t params[2] = {N, C};
    dev.main_queue().upload(*cache.sm_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &logits.buffer()},
        {1u, &target.buffer()},
        {2u, &loss.buffer()},
        {3u, cache.sm_params.get()},
    };
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 255u) / 256u);
    dev.main_queue().dispatch(*kernel, gx, 1, 1, bindings);
    return loss;
}

Tensor relu_backward(Device& dev, const Tensor& grad_out, const Tensor& x) {
    Tensor y(dev, x.shape());
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.relu_backward, kReluBackwardSrc);

    const std::vector<BufferBinding> bindings = {
        {0u, &grad_out.buffer()}, {1u, &x.buffer()}, {2u, &y.buffer()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((x.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return y;
}

Tensor cross_entropy_backward(Device& dev, const Tensor& logits, const Tensor& target) {
    const std::int32_t N = static_cast<std::int32_t>(logits.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(logits.dim(1));

    Tensor grad(dev, {N, C});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.cross_entropy_backward, kCrossEntropyBackwardSrc);
    if (!cache.sm_params) {
        cache.sm_params = dev.alloc(2 * sizeof(std::int32_t));
    }
    const std::int32_t params[2] = {N, C};
    dev.main_queue().upload(*cache.sm_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &logits.buffer()},
        {1u, &target.buffer()},
        {2u, &grad.buffer()},
        {3u, cache.sm_params.get()},
    };
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return grad;
}

Tensor transpose(Device& dev, const Tensor& x) {
    const std::int32_t M = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t N = static_cast<std::int32_t>(x.dim(1));

    Tensor y(dev, {N, M});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.transpose, kTransposeSrc);
    if (!cache.sm_params) {
        cache.sm_params = dev.alloc(2 * sizeof(std::int32_t));
    }
    const std::int32_t params[2] = {M, N};
    dev.main_queue().upload(*cache.sm_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &y.buffer()}, {2u, cache.sm_params.get()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((M * N + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return y;
}

Tensor sum_axis0(Device& dev, const Tensor& x) {
    const std::int32_t M = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t N = static_cast<std::int32_t>(x.dim(1));

    Tensor y(dev, {N});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.sum_axis0, kSumAxis0Src);
    if (!cache.sm_params) {
        cache.sm_params = dev.alloc(2 * sizeof(std::int32_t));
    }
    const std::int32_t params[2] = {M, N};
    dev.main_queue().upload(*cache.sm_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &y.buffer()}, {2u, cache.sm_params.get()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((N + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
    return y;
}

void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    Device& dev = param.device();
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.sgd_update, kSgdUpdateSrc);
    if (!cache.sgd_params) {
        cache.sgd_params = dev.alloc(sizeof(float));
    }
    dev.main_queue().upload(*cache.sgd_params, &lr, sizeof(lr));

    const std::vector<BufferBinding> bindings = {
        {0u, &param.buffer()}, {1u, &grad.buffer()}, {2u, cache.sgd_params.get()}};
    const std::uint32_t gx = static_cast<std::uint32_t>((param.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
}

Tensor conv2d_grad_input(Device& dev, const Tensor& grad_out, const Tensor& x,
                         const Tensor& w, int stride, int padding) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t Cin = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t Cout = static_cast<std::int32_t>(w.dim(0));
    const std::int32_t KH = static_cast<std::int32_t>(w.dim(2));
    const std::int32_t KW = static_cast<std::int32_t>(w.dim(3));

    // grad_col = W^T @ grad_Y^T = matmul(transpose(W), transpose(grad_Y))
    Tensor grad_Y = grad_out_reshape(dev, grad_out);                 // [Ncol, Cout]
    Tensor w_view = w.view({Cout, Cin * KH * KW});                   // [Cout, Kcol]
    Tensor w_T = transpose(dev, w_view);                             // [Kcol, Cout]
    Tensor grad_Y_T = transpose(dev, grad_Y);                        // [Cout, Ncol]
    Tensor grad_col = matmul(dev, w_T, grad_Y_T);                    // [Kcol, Ncol]
    return col2im(dev, grad_col, N, Cin, H, W, KH, KW, stride, padding);
}

Tensor conv2d_grad_weight(Device& dev, const Tensor& grad_out, const Tensor& x,
                          const Tensor& w, int stride, int padding) {
    const std::int32_t Cin = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t Cout = static_cast<std::int32_t>(w.dim(0));
    const std::int32_t KH = static_cast<std::int32_t>(w.dim(2));
    const std::int32_t KW = static_cast<std::int32_t>(w.dim(3));

    // grad_W^T = col @ grad_Y = matmul(im2col(x), grad_out_reshape(grad_out))
    Tensor col = im2col(dev, x, KH, KW, stride, padding);            // [Kcol, Ncol]
    Tensor grad_Y = grad_out_reshape(dev, grad_out);                 // [Ncol, Cout]
    Tensor grad_W_T = matmul(dev, col, grad_Y);                      // [Kcol, Cout]
    Tensor grad_W = transpose(dev, grad_W_T);                        // [Cout, Kcol]
    return grad_W.view({Cout, Cin, KH, KW});
}

Tensor conv2d_grad_bias(Device& dev, const Tensor& grad_out) {
    const std::int32_t N = static_cast<std::int32_t>(grad_out.dim(0));
    const std::int32_t Cout = static_cast<std::int32_t>(grad_out.dim(1));
    const std::int32_t Hout = static_cast<std::int32_t>(grad_out.dim(2));
    const std::int32_t Wout = static_cast<std::int32_t>(grad_out.dim(3));

    Tensor out(dev, {Cout});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.conv2d_grad_bias, kConv2dGradBiasSrc);
    if (!cache.conv_bias_params) {
        cache.conv_bias_params = dev.alloc(4 * sizeof(std::int32_t));
    }
    const std::int32_t params[4] = {N, Cout, Hout, Wout};
    dev.main_queue().upload(*cache.conv_bias_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &grad_out.buffer()}, {1u, &out.buffer()}, {2u, cache.conv_bias_params.get()}};
    const std::uint32_t wg = (static_cast<std::uint32_t>(Cout) + 255u) / 256u;
    dev.main_queue().dispatch(*k, wg, 1, 1, bindings);
    return out;
}

Tensor batchnorm_backward(Device& dev, const Tensor& x, const Tensor& grad_out,
                          const Tensor& gamma, const Tensor& mean, const Tensor& var,
                          float eps, Tensor& grad_gamma, Tensor& grad_beta) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t HW = H * W;

    Tensor out(dev, x.shape());
    auto& cache = cache_for(dev);

    if (!cache.bn_bwd_params) {
        cache.bn_bwd_params = dev.alloc(3 * sizeof(std::int32_t) + sizeof(float));
    }
    if (!cache.bn_stats_buf) {
        cache.bn_stats_buf = dev.alloc(2 * static_cast<std::size_t>(C) * sizeof(float));
    }
    struct BNParams {
        std::int32_t N;
        std::int32_t C;
        std::int32_t HW;
        float eps;
    };
    const BNParams p{N, C, HW, eps};
    dev.main_queue().upload(*cache.bn_bwd_params, &p, sizeof(p));

    // 第一步：统计 + 累加 grad_gamma/grad_beta
    Kernel* k_stats = get_or_compile(dev, cache.bn_backward_stats, kBnBackwardStatsSrc);
    {
        const std::vector<BufferBinding> bindings = {
            {0u, &grad_out.buffer()}, {1u, &x.buffer()}, {2u, &mean.buffer()},
            {3u, &var.buffer()},      {4u, &grad_gamma.buffer()},
            {5u, &grad_beta.buffer()}, {6u, cache.bn_stats_buf.get()},
            {7u, cache.bn_bwd_params.get()},
        };
        const std::uint32_t wg = static_cast<std::uint32_t>(C);  // 每 channel 一个 workgroup
        dev.main_queue().dispatch(*k_stats, wg, 1, 1, bindings);
    }

    // 第二步：计算 grad_x
    Kernel* k_gradx = get_or_compile(dev, cache.bn_backward_gradx, kBnBackwardGradXSrc);
    {
        const std::vector<BufferBinding> bindings = {
            {0u, &grad_out.buffer()}, {1u, &x.buffer()},   {2u, &mean.buffer()},
            {3u, &var.buffer()},      {4u, &gamma.buffer()}, {5u, cache.bn_stats_buf.get()},
            {6u, &out.buffer()},      {7u, cache.bn_bwd_params.get()},
        };
        const std::uint32_t total = static_cast<std::uint32_t>(N) * C * HW;
        const std::uint32_t wg = (total + 255u) / 256u;
        dev.main_queue().dispatch(*k_gradx, wg, 1, 1, bindings);
    }

    return out;
}

void bn_forward_stats(Device& dev, const Tensor& x, Tensor& mean, Tensor& var) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t C = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t HW = H * W;

    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.bn_fwd_stats, kBnForwardStatsSrc);
    if (!cache.bn_fwd_params) {
        cache.bn_fwd_params = dev.alloc(3 * sizeof(std::int32_t));
    }
    const std::int32_t params[3] = {N, C, HW};
    dev.main_queue().upload(*cache.bn_fwd_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()},
        {1u, &mean.buffer()},
        {2u, &var.buffer()},
        {3u, cache.bn_fwd_params.get()},
    };
    const std::uint32_t wg = static_cast<std::uint32_t>(C);  // 每 channel 一个 workgroup
    dev.main_queue().dispatch(*k, wg, 1, 1, bindings);
}

void bn_update_running_stats(Device& dev, Tensor& running_mean, Tensor& running_var,
                             const Tensor& batch_mean, const Tensor& batch_var,
                             float momentum) {
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.bn_update_running, kBnUpdateRunningSrc);
    if (!cache.bn_running_params) {
        cache.bn_running_params = dev.alloc(sizeof(float));
    }
    dev.main_queue().upload(*cache.bn_running_params, &momentum, sizeof(momentum));

    const std::vector<BufferBinding> bindings = {
        {0u, &running_mean.buffer()},
        {1u, &running_var.buffer()},
        {2u, &batch_mean.buffer()},
        {3u, &batch_var.buffer()},
        {4u, cache.bn_running_params.get()},
    };
    const std::uint32_t gx =
        static_cast<std::uint32_t>((running_mean.numel() + 255u) / 256u);
    dev.main_queue().dispatch(*k, gx, 1, 1, bindings);
}

Tensor col2im(Device& dev, const Tensor& col, int N, int Cin, int H, int W,
              int KH, int KW, int stride, int padding) {
    const std::int32_t Hout = (H + 2 * padding - KH) / stride + 1;
    const std::int32_t Wout = (W + 2 * padding - KW) / stride + 1;

    Tensor out(dev, {N, Cin, H, W});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.col2im, kCol2imSrc);
    if (!cache.col2im_params) {
        cache.col2im_params = dev.alloc(11 * sizeof(std::int32_t));
    }
    const std::int32_t params[11] = {N, Cin, H, W, Hout, Wout, KH, KW, stride, padding};
    dev.main_queue().upload(*cache.col2im_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &col.buffer()}, {1u, &out.buffer()}, {2u, cache.col2im_params.get()}};
    const std::uint32_t total = static_cast<std::uint32_t>(N) * Cin * H * W;
    const std::uint32_t wg = (total + 255u) / 256u;
    dev.main_queue().dispatch(*k, wg, 1, 1, bindings);
    return out;
}

Tensor grad_out_reshape(Device& dev, const Tensor& grad_out) {
    const std::int32_t N = static_cast<std::int32_t>(grad_out.dim(0));
    const std::int32_t Cout = static_cast<std::int32_t>(grad_out.dim(1));
    const std::int32_t Hout = static_cast<std::int32_t>(grad_out.dim(2));
    const std::int32_t Wout = static_cast<std::int32_t>(grad_out.dim(3));
    const std::int32_t Ncol = N * Hout * Wout;

    Tensor out(dev, {Ncol, Cout});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.grad_out_reshape, kGradOutReshapeSrc);
    if (!cache.grad_reshape_params) {
        cache.grad_reshape_params = dev.alloc(4 * sizeof(std::int32_t));
    }
    const std::int32_t params[4] = {Cout, Hout, Wout, Ncol};
    dev.main_queue().upload(*cache.grad_reshape_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &grad_out.buffer()}, {1u, &out.buffer()}, {2u, cache.grad_reshape_params.get()}};
    const std::uint32_t total = static_cast<std::uint32_t>(Ncol) * Cout;
    const std::uint32_t wg = (total + 255u) / 256u;
    dev.main_queue().dispatch(*k, wg, 1, 1, bindings);
    return out;
}

Tensor im2col(Device& dev, const Tensor& x, int KH, int KW, int stride, int padding) {
    const std::int32_t N = static_cast<std::int32_t>(x.dim(0));
    const std::int32_t Cin = static_cast<std::int32_t>(x.dim(1));
    const std::int32_t H = static_cast<std::int32_t>(x.dim(2));
    const std::int32_t W = static_cast<std::int32_t>(x.dim(3));
    const std::int32_t Hout = (H + 2 * padding - KH) / stride + 1;
    const std::int32_t Wout = (W + 2 * padding - KW) / stride + 1;
    const std::int32_t Kcol = Cin * KH * KW;
    const std::int32_t Ncol = N * Hout * Wout;

    Tensor col(dev, {Kcol, Ncol});
    auto& cache = cache_for(dev);
    Kernel* k = get_or_compile(dev, cache.im2col, kIm2colSrc);
    if (!cache.im2col_params) {
        cache.im2col_params = dev.alloc(10 * sizeof(std::int32_t));
    }
    const std::int32_t params[10] = {N, Cin, H, W, Hout, Wout, KH, KW, stride, padding};
    dev.main_queue().upload(*cache.im2col_params, params, sizeof(params));

    const std::vector<BufferBinding> bindings = {
        {0u, &x.buffer()}, {1u, &col.buffer()}, {2u, cache.im2col_params.get()}};
    const std::uint32_t total = static_cast<std::uint32_t>(Kcol) * Ncol;
    const std::uint32_t wg = (total + 255u) / 256u;
    dev.main_queue().dispatch(*k, wg, 1, 1, bindings);
    return col;
}

}  // namespace opendll::gl_ops
