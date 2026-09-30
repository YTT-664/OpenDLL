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

    std::cout << (all_ok ? "ALL OPS PASSED" : "SOME OPS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
