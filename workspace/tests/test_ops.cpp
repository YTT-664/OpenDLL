#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "opendll/device.hpp"
#include "opendll/ops.hpp"
#include "opendll/tensor.hpp"

using namespace opendll;

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static std::vector<float> random_floats(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) {
        x = dist(rng);
    }
    return v;
}

int main() {
    auto cpu = Device::create(Backend::CPU);
    auto gl = Device::create(Backend::OpenGL);
    if (!cpu) {
        std::cerr << "[test_ops] CPU device creation failed\n";
        return 1;
    }
    if (!gl) {
        std::cerr << "[test_ops] OpenGL device creation failed\n";
        return 2;
    }

    std::mt19937 rng(42);
    bool all_ok = true;

    // ---- matmul ----
    {
        const int64_t M = 64, K = 32, N = 48;
        const auto A = random_floats(M * K, rng);
        const auto B = random_floats(K * N, rng);

        Tensor ta_cpu(*cpu, {M, K}), tb_cpu(*cpu, {K, N});
        ta_cpu.upload(A);
        tb_cpu.upload(B);
        Tensor tc_cpu = matmul(*cpu, ta_cpu, tb_cpu);

        Tensor ta_gl(*gl, {M, K}), tb_gl(*gl, {K, N});
        ta_gl.upload(A);
        tb_gl.upload(B);
        Tensor tc_gl = matmul(*gl, ta_gl, tb_gl);
        gl->synchronize();

        std::vector<float> c_cpu, c_gl;
        tc_cpu.download(c_cpu);
        tc_gl.download(c_gl);
        const float d = max_abs_diff(c_cpu, c_gl);
        const bool ok = d < 1e-3f;
        std::cout << "[matmul] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- relu ----
    {
        const int64_t N = 1024;
        const auto X = random_floats(N, rng);

        Tensor tx_cpu(*cpu, {N});
        tx_cpu.upload(X);
        Tensor ty_cpu = relu(*cpu, tx_cpu);

        Tensor tx_gl(*gl, {N});
        tx_gl.upload(X);
        Tensor ty_gl = relu(*gl, tx_gl);
        gl->synchronize();

        std::vector<float> y_cpu, y_gl;
        ty_cpu.download(y_cpu);
        ty_gl.download(y_gl);
        const float d = max_abs_diff(y_cpu, y_gl);
        const bool ok = d < 1e-4f;
        std::cout << "[relu] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- add ----
    {
        const int64_t N = 1024;
        const auto A = random_floats(N, rng);
        const auto B = random_floats(N, rng);

        Tensor ta_cpu(*cpu, {N}), tb_cpu(*cpu, {N});
        ta_cpu.upload(A);
        tb_cpu.upload(B);
        Tensor tc_cpu = add(*cpu, ta_cpu, tb_cpu);

        Tensor ta_gl(*gl, {N}), tb_gl(*gl, {N});
        ta_gl.upload(A);
        tb_gl.upload(B);
        Tensor tc_gl = add(*gl, ta_gl, tb_gl);
        gl->synchronize();

        std::vector<float> c_cpu, c_gl;
        tc_cpu.download(c_cpu);
        tc_gl.download(c_gl);
        const float d = max_abs_diff(c_cpu, c_gl);
        const bool ok = d < 1e-4f;
        std::cout << "[add] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- mul ----
    {
        const int64_t N = 1024;
        const auto A = random_floats(N, rng);
        const auto B = random_floats(N, rng);

        Tensor ta_cpu(*cpu, {N}), tb_cpu(*cpu, {N});
        ta_cpu.upload(A);
        tb_cpu.upload(B);
        Tensor tc_cpu = mul(*cpu, ta_cpu, tb_cpu);

        Tensor ta_gl(*gl, {N}), tb_gl(*gl, {N});
        ta_gl.upload(A);
        tb_gl.upload(B);
        Tensor tc_gl = mul(*gl, ta_gl, tb_gl);
        gl->synchronize();

        std::vector<float> c_cpu, c_gl;
        tc_cpu.download(c_cpu);
        tc_gl.download(c_gl);
        const float d = max_abs_diff(c_cpu, c_gl);
        const bool ok = d < 1e-4f;
        std::cout << "[mul] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- conv2d ----
    {
        const int64_t N = 1, Cin = 3, H = 16, W = 16;
        const int64_t Cout = 4, KH = 3, KW = 3;
        const int stride = 1, padding = 1;

        const auto X = random_floats(static_cast<std::size_t>(N * Cin * H * W), rng);
        const auto Wt = random_floats(static_cast<std::size_t>(Cout * Cin * KH * KW), rng);
        const auto B = random_floats(static_cast<std::size_t>(Cout), rng);

        Tensor x_cpu(*cpu, {N, Cin, H, W});
        Tensor w_cpu(*cpu, {Cout, Cin, KH, KW});
        Tensor b_cpu(*cpu, {Cout});
        x_cpu.upload(X);
        w_cpu.upload(Wt);
        b_cpu.upload(B);
        Tensor y_cpu = conv2d(*cpu, x_cpu, w_cpu, b_cpu, stride, padding);

        Tensor x_gl(*gl, {N, Cin, H, W});
        Tensor w_gl(*gl, {Cout, Cin, KH, KW});
        Tensor b_gl(*gl, {Cout});
        x_gl.upload(X);
        w_gl.upload(Wt);
        b_gl.upload(B);
        Tensor y_gl = conv2d(*gl, x_gl, w_gl, b_gl, stride, padding);
        gl->synchronize();

        std::vector<float> out_cpu, out_gl;
        y_cpu.download(out_cpu);
        y_gl.download(out_gl);
        const float d = max_abs_diff(out_cpu, out_gl);
        const bool ok = d < 1e-3f;
        std::cout << "[conv2d] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- linear ----
    {
        const int64_t M = 64, K = 128, N = 32;
        const auto X = random_floats(static_cast<std::size_t>(M * K), rng);
        const auto Wt = random_floats(static_cast<std::size_t>(N * K), rng);
        const auto B = random_floats(static_cast<std::size_t>(N), rng);

        Tensor x_cpu(*cpu, {M, K});
        Tensor w_cpu(*cpu, {N, K});
        Tensor b_cpu(*cpu, {N});
        x_cpu.upload(X);
        w_cpu.upload(Wt);
        b_cpu.upload(B);
        Tensor y_cpu = linear(*cpu, x_cpu, w_cpu, b_cpu);

        Tensor x_gl(*gl, {M, K});
        Tensor w_gl(*gl, {N, K});
        Tensor b_gl(*gl, {N});
        x_gl.upload(X);
        w_gl.upload(Wt);
        b_gl.upload(B);
        Tensor y_gl = linear(*gl, x_gl, w_gl, b_gl);
        gl->synchronize();

        std::vector<float> out_cpu, out_gl;
        y_cpu.download(out_cpu);
        y_gl.download(out_gl);
        const float d = max_abs_diff(out_cpu, out_gl);
        const bool ok = d < 1e-3f;
        std::cout << "[linear] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- maxpool2d ----
    {
        const int64_t N = 1, C = 2, H = 8, W = 8;
        const int kernel = 3, stride = 2, padding = 1;
        const auto X = random_floats(static_cast<std::size_t>(N * C * H * W), rng);

        Tensor x_cpu(*cpu, {N, C, H, W});
        x_cpu.upload(X);
        Tensor y_cpu = maxpool2d(*cpu, x_cpu, kernel, stride, padding);

        Tensor x_gl(*gl, {N, C, H, W});
        x_gl.upload(X);
        Tensor y_gl = maxpool2d(*gl, x_gl, kernel, stride, padding);
        gl->synchronize();

        std::vector<float> a, b;
        y_cpu.download(a);
        y_gl.download(b);
        const float d = max_abs_diff(a, b);
        const bool ok = d < 1e-4f;
        std::cout << "[maxpool2d] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- avgpool2d ----
    {
        const int64_t N = 1, C = 2, H = 8, W = 8;
        const int kernel = 2, stride = 2, padding = 0;
        const auto X = random_floats(static_cast<std::size_t>(N * C * H * W), rng);

        Tensor x_cpu(*cpu, {N, C, H, W});
        x_cpu.upload(X);
        Tensor y_cpu = avgpool2d(*cpu, x_cpu, kernel, stride, padding);

        Tensor x_gl(*gl, {N, C, H, W});
        x_gl.upload(X);
        Tensor y_gl = avgpool2d(*gl, x_gl, kernel, stride, padding);
        gl->synchronize();

        std::vector<float> a, b;
        y_cpu.download(a);
        y_gl.download(b);
        const float d = max_abs_diff(a, b);
        const bool ok = d < 1e-4f;
        std::cout << "[avgpool2d] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- batchnorm2d ----
    {
        const int64_t N = 2, C = 3, H = 4, W = 4;
        const float eps = 1e-5f;
        const auto X = random_floats(static_cast<std::size_t>(N * C * H * W), rng);
        const auto G = random_floats(static_cast<std::size_t>(C), rng);
        const auto B = random_floats(static_cast<std::size_t>(C), rng);
        const auto M = random_floats(static_cast<std::size_t>(C), rng);
        std::vector<float> V(C);
        for (int i = 0; i < C; ++i) {
            V[i] = 0.5f + 0.2f * i;
        }

        Tensor x_cpu(*cpu, {N, C, H, W}), g_cpu(*cpu, {C}), b_cpu(*cpu, {C}),
            m_cpu(*cpu, {C}), v_cpu(*cpu, {C});
        x_cpu.upload(X);
        g_cpu.upload(G);
        b_cpu.upload(B);
        m_cpu.upload(M);
        v_cpu.upload(V);
        Tensor y_cpu = batchnorm2d(*cpu, x_cpu, g_cpu, b_cpu, m_cpu, v_cpu, eps);

        Tensor x_gl(*gl, {N, C, H, W}), g_gl(*gl, {C}), b_gl(*gl, {C}),
            m_gl(*gl, {C}), v_gl(*gl, {C});
        x_gl.upload(X);
        g_gl.upload(G);
        b_gl.upload(B);
        m_gl.upload(M);
        v_gl.upload(V);
        Tensor y_gl = batchnorm2d(*gl, x_gl, g_gl, b_gl, m_gl, v_gl, eps);
        gl->synchronize();

        std::vector<float> a, b;
        y_cpu.download(a);
        y_gl.download(b);
        const float d = max_abs_diff(a, b);
        const bool ok = d < 1e-3f;
        std::cout << "[batchnorm2d] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- softmax ----
    {
        const int64_t N = 8, C = 10;
        const auto X = random_floats(static_cast<std::size_t>(N * C), rng);

        Tensor x_cpu(*cpu, {N, C});
        x_cpu.upload(X);
        Tensor y_cpu = softmax(*cpu, x_cpu);

        Tensor x_gl(*gl, {N, C});
        x_gl.upload(X);
        Tensor y_gl = softmax(*gl, x_gl);
        gl->synchronize();

        std::vector<float> a, b;
        y_cpu.download(a);
        y_gl.download(b);
        const float d = max_abs_diff(a, b);
        const bool ok = d < 1e-3f;
        std::cout << "[softmax] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    // ---- cross_entropy ----
    {
        const int64_t N = 8, C = 10;
        const auto X = random_floats(static_cast<std::size_t>(N * C), rng);
        std::uniform_int_distribution<int> label_dist(0, static_cast<int>(C - 1));
        std::vector<float> T(N);
        for (auto& t : T) {
            t = static_cast<float>(label_dist(rng));
        }

        Tensor l_cpu(*cpu, {N, C}), t_cpu(*cpu, {N});
        l_cpu.upload(X);
        t_cpu.upload(T);
        Tensor loss_cpu = cross_entropy(*cpu, l_cpu, t_cpu);

        Tensor l_gl(*gl, {N, C}), t_gl(*gl, {N});
        l_gl.upload(X);
        t_gl.upload(T);
        Tensor loss_gl = cross_entropy(*gl, l_gl, t_gl);
        gl->synchronize();

        std::vector<float> a, b;
        loss_cpu.download(a);
        loss_gl.download(b);
        const float d = max_abs_diff(a, b);
        const bool ok = d < 1e-3f;
        std::cout << "[cross_entropy] max_abs_diff=" << d << (ok ? " PASS" : " FAIL") << "\n";
        all_ok = all_ok && ok;
    }

    std::cout << (all_ok ? "ALL OPS PASSED" : "SOME OPS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
