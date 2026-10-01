#include "cpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int Cin = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Cout = static_cast<int>(w.dim(0));
    const int KH = static_cast<int>(w.dim(2));
    const int KW = static_cast<int>(w.dim(3));
    const int Hout = (H + 2 * padding - KH) / stride + 1;
    const int Wout = (W + 2 * padding - KW) / stride + 1;

    std::vector<float> X, Wt, B;
    x.download(X);
    w.download(Wt);
    b.download(B);

    std::vector<float> Y(static_cast<std::size_t>(N) * Cout * Hout * Wout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float sum = B[co];
                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < KH; ++kh) {
                            const int ih = oh * stride - padding + kh;
                            if (ih < 0 || ih >= H) {
                                continue;
                            }
                            for (int kw = 0; kw < KW; ++kw) {
                                const int iw = ow * stride - padding + kw;
                                if (iw < 0 || iw >= W) {
                                    continue;
                                }
                                const std::size_t xi =
                                    ((static_cast<std::size_t>(n) * Cin + ci) * H + ih) * W + iw;
                                const std::size_t wi =
                                    ((static_cast<std::size_t>(co) * Cin + ci) * KH + kh) * KW + kw;
                                sum += X[xi] * Wt[wi];
                            }
                        }
                    }
                    const std::size_t yi =
                        ((static_cast<std::size_t>(n) * Cout + co) * Hout + oh) * Wout + ow;
                    Y[yi] = sum;
                }
            }
        }
    }

    Tensor out(dev, {N, Cout, Hout, Wout});
    out.upload(Y);
    return out;
}

Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b) {
    const std::size_t M = static_cast<std::size_t>(x.dim(0));
    const std::size_t K = static_cast<std::size_t>(x.dim(1));
    const std::size_t N = static_cast<std::size_t>(w.dim(0));

    std::vector<float> X, Wt, B;
    x.download(X);
    w.download(Wt);
    b.download(B);

    std::vector<float> Y(M * N, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            float sum = B[n];
            for (std::size_t k = 0; k < K; ++k) {
                sum += X[m * K + k] * Wt[n * K + k];
            }
            Y[m * N + n] = sum;
        }
    }

    Tensor out(dev, {static_cast<int64_t>(M), static_cast<int64_t>(N)});
    out.upload(Y);
    return out;
}

Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Hout = (H + 2 * padding - kernel) / stride + 1;
    const int Wout = (W + 2 * padding - kernel) / stride + 1;

    std::vector<float> X;
    x.download(X);
    std::vector<float> Y(static_cast<std::size_t>(N) * C * Hout * Wout,
                         -std::numeric_limits<float>::infinity());

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float m = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel; ++kh) {
                        const int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= H) {
                            continue;
                        }
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int iw = ow * stride - padding + kw;
                            if (iw < 0 || iw >= W) {
                                continue;
                            }
                            m = std::max(m, X[((static_cast<std::size_t>(n) * C + c) * H + ih) * W + iw]);
                        }
                    }
                    Y[((static_cast<std::size_t>(n) * C + c) * Hout + oh) * Wout + ow] = m;
                }
            }
        }
    }

    Tensor out(dev, {N, C, Hout, Wout});
    out.upload(Y);
    return out;
}

Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Hout = (H + 2 * padding - kernel) / stride + 1;
    const int Wout = (W + 2 * padding - kernel) / stride + 1;

    std::vector<float> X;
    x.download(X);
    std::vector<float> Y(static_cast<std::size_t>(N) * C * Hout * Wout, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    float sum = 0.0f;
                    int cnt = 0;
                    for (int kh = 0; kh < kernel; ++kh) {
                        const int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= H) {
                            continue;
                        }
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int iw = ow * stride - padding + kw;
                            if (iw < 0 || iw >= W) {
                                continue;
                            }
                            sum += X[((static_cast<std::size_t>(n) * C + c) * H + ih) * W + iw];
                            ++cnt;
                        }
                    }
                    Y[((static_cast<std::size_t>(n) * C + c) * Hout + oh) * Wout + ow] =
                        sum / static_cast<float>(cnt);
                }
            }
        }
    }

    Tensor out(dev, {N, C, Hout, Wout});
    out.upload(Y);
    return out;
}

Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int HW = H * W;

    std::vector<float> X, G, B, M, V;
    x.download(X);
    gamma.download(G);
    beta.download(B);
    mean.download(M);
    var.download(V);

    std::vector<float> Y(X.size());
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const float scale = 1.0f / std::sqrt(V[c] + eps);
            for (int i = 0; i < HW; ++i) {
                const std::size_t idx = (static_cast<std::size_t>(n) * C + c) * HW + i;
                Y[idx] = (X[idx] - M[c]) * scale * G[c] + B[c];
            }
        }
    }

    Tensor out(dev, x.shape());
    out.upload(Y);
    return out;
}

Tensor softmax(Device& dev, const Tensor& x) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));

    std::vector<float> X;
    x.download(X);
    std::vector<float> Y(static_cast<std::size_t>(N) * C);

    for (int i = 0; i < N; ++i) {
        float m = X[static_cast<std::size_t>(i) * C];
        for (int j = 1; j < C; ++j) {
            m = std::max(m, X[static_cast<std::size_t>(i) * C + j]);
        }
        float s = 0.0f;
        for (int j = 0; j < C; ++j) {
            const float e = std::exp(X[static_cast<std::size_t>(i) * C + j] - m);
            Y[static_cast<std::size_t>(i) * C + j] = e;
            s += e;
        }
        for (int j = 0; j < C; ++j) {
            Y[static_cast<std::size_t>(i) * C + j] /= s;
        }
    }

    Tensor out(dev, {N, C});
    out.upload(Y);
    return out;
}

Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target) {
    const int N = static_cast<int>(logits.dim(0));
    const int C = static_cast<int>(logits.dim(1));

    std::vector<float> L, T;
    logits.download(L);
    target.download(T);

    std::vector<float> loss(N);
    for (int i = 0; i < N; ++i) {
        float m = L[static_cast<std::size_t>(i) * C];
        for (int j = 1; j < C; ++j) {
            m = std::max(m, L[static_cast<std::size_t>(i) * C + j]);
        }
        float s = 0.0f;
        for (int j = 0; j < C; ++j) {
            s += std::exp(L[static_cast<std::size_t>(i) * C + j] - m);
        }
        const float logsumexp = m + std::log(s);
        const int t = static_cast<int>(T[i]);
        loss[i] = logsumexp - L[static_cast<std::size_t>(i) * C + t];
    }

    Tensor out(dev, {N});
    out.upload(loss);
    return out;
}

Tensor relu_backward(Device& dev, const Tensor& grad_out, const Tensor& x) {
    std::vector<float> GO, X;
    grad_out.download(GO);
    x.download(X);
    for (std::size_t i = 0; i < X.size(); ++i) {
        GO[i] = X[i] > 0.0f ? GO[i] : 0.0f;
    }
    Tensor out(dev, x.shape());
    out.upload(GO);
    return out;
}

Tensor cross_entropy_backward(Device& dev, const Tensor& logits, const Tensor& target) {
    const int N = static_cast<int>(logits.dim(0));
    const int C = static_cast<int>(logits.dim(1));

    std::vector<float> L, T;
    logits.download(L);
    target.download(T);

    std::vector<float> G(static_cast<std::size_t>(N) * C);
    for (int i = 0; i < N; ++i) {
        float m = L[static_cast<std::size_t>(i) * C];
        for (int j = 1; j < C; ++j) {
            m = std::max(m, L[static_cast<std::size_t>(i) * C + j]);
        }
        float s = 0.0f;
        for (int j = 0; j < C; ++j) {
            s += std::exp(L[static_cast<std::size_t>(i) * C + j] - m);
        }
        const int t = static_cast<int>(T[i]);
        for (int j = 0; j < C; ++j) {
            const float p = std::exp(L[static_cast<std::size_t>(i) * C + j] - m) / s;
            G[static_cast<std::size_t>(i) * C + j] = p - (j == t ? 1.0f : 0.0f);
        }
    }

    Tensor out(dev, {N, C});
    out.upload(G);
    return out;
}

Tensor transpose(Device& dev, const Tensor& x) {
    const int M = static_cast<int>(x.dim(0));
    const int N = static_cast<int>(x.dim(1));

    std::vector<float> X;
    x.download(X);
    std::vector<float> Y(static_cast<std::size_t>(M) * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Y[static_cast<std::size_t>(j) * M + i] = X[static_cast<std::size_t>(i) * N + j];
        }
    }

    Tensor out(dev, {N, M});
    out.upload(Y);
    return out;
}

Tensor sum_axis0(Device& dev, const Tensor& x) {
    const int M = static_cast<int>(x.dim(0));
    const int N = static_cast<int>(x.dim(1));

    std::vector<float> X;
    x.download(X);
    std::vector<float> Y(N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Y[j] += X[static_cast<std::size_t>(i) * N + j];
        }
    }

    Tensor out(dev, {N});
    out.upload(Y);
    return out;
}

void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    std::vector<float> p, g;
    param.download(p);
    grad.download(g);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p[i] -= lr * g[i];
    }
    param.upload(p);
}

void bn_update_running_stats(Device& dev, Tensor& running_mean, Tensor& running_var,
                             const Tensor& batch_mean, const Tensor& batch_var,
                             float momentum) {
    std::vector<float> rm, rv, bm, bv;
    running_mean.download(rm);
    running_var.download(rv);
    batch_mean.download(bm);
    batch_var.download(bv);
    for (std::size_t i = 0; i < rm.size(); ++i) {
        rm[i] = (1.0f - momentum) * rm[i] + momentum * bm[i];
        rv[i] = (1.0f - momentum) * rv[i] + momentum * bv[i];
    }
    running_mean.upload(rm);
    running_var.upload(rv);
}

Tensor conv2d_grad_input(Device& dev, const Tensor& grad_out, const Tensor& x,
                         const Tensor& w, int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int Cin = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Cout = static_cast<int>(w.dim(0));
    const int KH = static_cast<int>(w.dim(2));
    const int KW = static_cast<int>(w.dim(3));
    const int Hout = static_cast<int>(grad_out.dim(2));
    const int Wout = static_cast<int>(grad_out.dim(3));

    std::vector<float> GO, Wt;
    grad_out.download(GO);
    w.download(Wt);

    std::vector<float> GX(static_cast<std::size_t>(N) * Cin * H * W, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int ci = 0; ci < Cin; ++ci) {
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    float sum = 0.0f;
                    for (int co = 0; co < Cout; ++co) {
                        for (int kh = 0; kh < KH; ++kh) {
                            const int num_h = ih + padding - kh;
                            if (num_h % stride != 0) {
                                continue;
                            }
                            const int oh = num_h / stride;
                            if (oh < 0 || oh >= Hout) {
                                continue;
                            }
                            for (int kw = 0; kw < KW; ++kw) {
                                const int num_w = iw + padding - kw;
                                if (num_w % stride != 0) {
                                    continue;
                                }
                                const int ow = num_w / stride;
                                if (ow < 0 || ow >= Wout) {
                                    continue;
                                }
                                sum += GO[((static_cast<std::size_t>(n) * Cout + co) * Hout + oh) *
                                              Wout +
                                          ow] *
                                       Wt[((static_cast<std::size_t>(co) * Cin + ci) * KH + kh) *
                                              KW +
                                          kw];
                            }
                        }
                    }
                    GX[((static_cast<std::size_t>(n) * Cin + ci) * H + ih) * W + iw] = sum;
                }
            }
        }
    }

    Tensor out(dev, {N, Cin, H, W});
    out.upload(GX);
    return out;
}

Tensor conv2d_grad_weight(Device& dev, const Tensor& grad_out, const Tensor& x,
                          const Tensor& w, int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int Cin = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Cout = static_cast<int>(w.dim(0));
    const int KH = static_cast<int>(w.dim(2));
    const int KW = static_cast<int>(w.dim(3));
    const int Hout = static_cast<int>(grad_out.dim(2));
    const int Wout = static_cast<int>(grad_out.dim(3));

    std::vector<float> GO, X;
    grad_out.download(GO);
    x.download(X);

    std::vector<float> GW(static_cast<std::size_t>(Cout) * Cin * KH * KW, 0.0f);
    for (int co = 0; co < Cout; ++co) {
        for (int ci = 0; ci < Cin; ++ci) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    float sum = 0.0f;
                    for (int n = 0; n < N; ++n) {
                        for (int oh = 0; oh < Hout; ++oh) {
                            const int ih = oh * stride - padding + kh;
                            if (ih < 0 || ih >= H) {
                                continue;
                            }
                            for (int ow = 0; ow < Wout; ++ow) {
                                const int iw = ow * stride - padding + kw;
                                if (iw < 0 || iw >= W) {
                                    continue;
                                }
                                sum += GO[((static_cast<std::size_t>(n) * Cout + co) * Hout + oh) *
                                              Wout +
                                          ow] *
                                       X[((static_cast<std::size_t>(n) * Cin + ci) * H + ih) * W +
                                         iw];
                            }
                        }
                    }
                    GW[((static_cast<std::size_t>(co) * Cin + ci) * KH + kh) * KW + kw] = sum;
                }
            }
        }
    }

    Tensor out(dev, {Cout, Cin, KH, KW});
    out.upload(GW);
    return out;
}

Tensor conv2d_grad_bias(Device& dev, const Tensor& grad_out) {
    const int N = static_cast<int>(grad_out.dim(0));
    const int Cout = static_cast<int>(grad_out.dim(1));
    const int Hout = static_cast<int>(grad_out.dim(2));
    const int Wout = static_cast<int>(grad_out.dim(3));

    std::vector<float> GO;
    grad_out.download(GO);
    std::vector<float> GB(Cout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int oh = 0; oh < Hout; ++oh) {
                for (int ow = 0; ow < Wout; ++ow) {
                    GB[co] += GO[((static_cast<std::size_t>(n) * Cout + co) * Hout + oh) * Wout +
                                 ow];
                }
            }
        }
    }

    Tensor out(dev, {Cout});
    out.upload(GB);
    return out;
}

Tensor batchnorm_backward(Device& dev, const Tensor& x, const Tensor& grad_out,
                          const Tensor& gamma, const Tensor& mean, const Tensor& var,
                          float eps, Tensor& grad_gamma, Tensor& grad_beta) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int HW = H * W;
    const float M = static_cast<float>(N) * HW;

    std::vector<float> X, GO, G, MEAN, VAR, GG, GB;
    x.download(X);
    grad_out.download(GO);
    gamma.download(G);
    mean.download(MEAN);
    var.download(VAR);
    grad_gamma.download(GG);
    grad_beta.download(GB);

    std::vector<float> GX(X.size(), 0.0f);

    for (int c = 0; c < C; ++c) {
        const float inv_std = 1.0f / std::sqrt(VAR[c] + eps);
        float sum_gy = 0.0f;
        float sum_gy_xhat = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < HW; ++i) {
                const std::size_t idx = (static_cast<std::size_t>(n) * C + c) * HW + i;
                const float xhat = (X[idx] - MEAN[c]) * inv_std;
                const float gy = GO[idx];
                sum_gy += gy;
                sum_gy_xhat += gy * xhat;
            }
        }
        GG[c] += sum_gy_xhat;
        GB[c] += sum_gy;

        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < HW; ++i) {
                const std::size_t idx = (static_cast<std::size_t>(n) * C + c) * HW + i;
                const float xhat = (X[idx] - MEAN[c]) * inv_std;
                const float gy = GO[idx];
                GX[idx] = inv_std * G[c] * (gy - sum_gy / M - xhat * sum_gy_xhat / M);
            }
        }
    }

    grad_gamma.upload(GG);
    grad_beta.upload(GB);

    Tensor out(dev, x.shape());
    out.upload(GX);
    return out;
}

void bn_forward_stats(Device& dev, const Tensor& x, Tensor& mean, Tensor& var) {
    const int N = static_cast<int>(x.dim(0));
    const int C = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int HW = H * W;
    const float M = static_cast<float>(N) * HW;

    std::vector<float> X;
    x.download(X);
    std::vector<float> MEAN(C), VAR(C);
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < HW; ++i) {
                const float v = X[(static_cast<std::size_t>(n) * C + c) * HW + i];
                sum += v;
                sum2 += v * v;
            }
        }
        const float mu = sum / M;
        MEAN[c] = mu;
        VAR[c] = sum2 / M - mu * mu;
    }
    mean.upload(MEAN);
    var.upload(VAR);
}

Tensor col2im(Device& dev, const Tensor& col, int N, int Cin, int H, int W,
              int KH, int KW, int stride, int padding) {
    const int Hout = (H + 2 * padding - KH) / stride + 1;
    const int Wout = (W + 2 * padding - KW) / stride + 1;
    const int Kcol = Cin * KH * KW;
    const int Ncol = N * Hout * Wout;

    std::vector<float> COL;
    col.download(COL);
    std::vector<float> GX(static_cast<std::size_t>(N) * Cin * H * W, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int ci = 0; ci < Cin; ++ci) {
            for (int ih = 0; ih < H; ++ih) {
                for (int iw = 0; iw < W; ++iw) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int num_h = ih + padding - kh;
                        if (num_h % stride != 0) {
                            continue;
                        }
                        const int oh = num_h / stride;
                        if (oh < 0 || oh >= Hout) {
                            continue;
                        }
                        for (int kw = 0; kw < KW; ++kw) {
                            const int num_w = iw + padding - kw;
                            if (num_w % stride != 0) {
                                continue;
                            }
                            const int ow = num_w / stride;
                            if (ow < 0 || ow >= Wout) {
                                continue;
                            }
                            const int p = (n * Hout + oh) * Wout + ow;
                            const int k = (ci * KH + kh) * KW + kw;
                            sum += COL[static_cast<std::size_t>(k) * Ncol + p];
                        }
                    }
                    GX[((static_cast<std::size_t>(n) * Cin + ci) * H + ih) * W + iw] = sum;
                }
            }
        }
    }

    Tensor out(dev, {N, Cin, H, W});
    out.upload(GX);
    return out;
}

Tensor grad_out_reshape(Device& dev, const Tensor& grad_out) {
    const int N = static_cast<int>(grad_out.dim(0));
    const int Cout = static_cast<int>(grad_out.dim(1));
    const int Hout = static_cast<int>(grad_out.dim(2));
    const int Wout = static_cast<int>(grad_out.dim(3));
    const int Ncol = N * Hout * Wout;

    std::vector<float> GO;
    grad_out.download(GO);
    std::vector<float> GY(static_cast<std::size_t>(Ncol) * Cout);
    for (int p = 0; p < Ncol; ++p) {
        const int ow = p % Wout;
        const int oh = (p / Wout) % Hout;
        const int n = p / (Wout * Hout);
        for (int co = 0; co < Cout; ++co) {
            GY[static_cast<std::size_t>(p) * Cout + co] =
                GO[((static_cast<std::size_t>(n) * Cout + co) * Hout + oh) * Wout + ow];
        }
    }

    Tensor out(dev, {Ncol, Cout});
    out.upload(GY);
    return out;
}

Tensor im2col(Device& dev, const Tensor& x, int KH, int KW, int stride, int padding) {
    const int N = static_cast<int>(x.dim(0));
    const int Cin = static_cast<int>(x.dim(1));
    const int H = static_cast<int>(x.dim(2));
    const int W = static_cast<int>(x.dim(3));
    const int Hout = (H + 2 * padding - KH) / stride + 1;
    const int Wout = (W + 2 * padding - KW) / stride + 1;
    const int Kcol = Cin * KH * KW;
    const int Ncol = N * Hout * Wout;

    std::vector<float> X;
    x.download(X);
    std::vector<float> COL(static_cast<std::size_t>(Kcol) * Ncol, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < Hout; ++oh) {
            for (int ow = 0; ow < Wout; ++ow) {
                const int p = (n * Hout + oh) * Wout + ow;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= H) {
                            continue;
                        }
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * stride - padding + kw;
                            if (iw < 0 || iw >= W) {
                                continue;
                            }
                            const int k = (ci * KH + kh) * KW + kw;
                            COL[static_cast<std::size_t>(k) * Ncol + p] =
                                X[((static_cast<std::size_t>(n) * Cin + ci) * H + ih) * W + iw];
                        }
                    }
                }
            }
        }
    }

    Tensor out(dev, {Kcol, Ncol});
    out.upload(COL);
    return out;
}

}  // namespace opendll::cpu_ops
