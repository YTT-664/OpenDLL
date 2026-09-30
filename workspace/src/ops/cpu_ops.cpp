#include "cpu_ops.hpp"

#include <cstddef>
#include <vector>

namespace opendll::cpu_ops {

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b) {
    const std::size_t M = static_cast<std::size_t>(a.dim(0));
    const std::size_t K = static_cast<std::size_t>(a.dim(1));
    const std::size_t N = static_cast<std::size_t>(b.dim(1));

    std::vector<float> A, B;
    a.download(A);
    b.download(B);

    std::vector<float> C(M * N, 0.0f);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const float aik = A[i * K + k];
            for (std::size_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    Tensor out(dev, {static_cast<int64_t>(M), static_cast<int64_t>(N)});
    out.upload(C);
    return out;
}

Tensor relu(Device& dev, const Tensor& x) {
    std::vector<float> X;
    x.download(X);
    for (float& v : X) {
        v = v > 0.0f ? v : 0.0f;
    }
    Tensor out(dev, x.shape());
    out.upload(X);
    return out;
}

Tensor add(Device& dev, const Tensor& a, const Tensor& b) {
    std::vector<float> A, B;
    a.download(A);
    b.download(B);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] += B[i];
    }
    Tensor out(dev, a.shape());
    out.upload(A);
    return out;
}

Tensor mul(Device& dev, const Tensor& a, const Tensor& b) {
    std::vector<float> A, B;
    a.download(A);
    b.download(B);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] *= B[i];
    }
    Tensor out(dev, a.shape());
    out.upload(A);
    return out;
}

}  // namespace opendll::cpu_ops
