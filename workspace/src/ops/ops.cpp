#include "opendll/ops.hpp"

#include <stdexcept>

#include "cpu_ops.hpp"

#ifdef OPENDLL_HAS_OPENGL
#include "gl_ops.hpp"
#endif

namespace opendll {

Tensor matmul(Device& dev, const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    if (a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmul shape mismatch");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::matmul(dev, a, b);
    }
#endif
    return cpu_ops::matmul(dev, a, b);
}

Tensor relu(Device& dev, const Tensor& x) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::relu(dev, x);
    }
#endif
    return cpu_ops::relu(dev, x);
}

Tensor add(Device& dev, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add shape mismatch");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::add(dev, a, b);
    }
#endif
    return cpu_ops::add(dev, a, b);
}

Tensor mul(Device& dev, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("mul shape mismatch");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::mul(dev, a, b);
    }
#endif
    return cpu_ops::mul(dev, a, b);
}

Tensor conv2d(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b,
              int stride, int padding) {
    if (x.ndim() != 4 || w.ndim() != 4) {
        throw std::invalid_argument("conv2d expects 4D x and w");
    }
    if (x.dim(1) != w.dim(1)) {
        throw std::invalid_argument("conv2d channel mismatch");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::conv2d(dev, x, w, b, stride, padding);
    }
#endif
    return cpu_ops::conv2d(dev, x, w, b, stride, padding);
}

Tensor linear(Device& dev, const Tensor& x, const Tensor& w, const Tensor& b) {
    if (x.ndim() != 2 || w.ndim() != 2) {
        throw std::invalid_argument("linear expects 2D x and w");
    }
    if (x.dim(1) != w.dim(1)) {
        throw std::invalid_argument("linear shape mismatch");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::linear(dev, x, w, b);
    }
#endif
    return cpu_ops::linear(dev, x, w, b);
}

Tensor maxpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    if (x.ndim() != 4) {
        throw std::invalid_argument("maxpool2d expects 4D x");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::maxpool2d(dev, x, kernel, stride, padding);
    }
#endif
    return cpu_ops::maxpool2d(dev, x, kernel, stride, padding);
}

Tensor avgpool2d(Device& dev, const Tensor& x, int kernel, int stride, int padding) {
    if (x.ndim() != 4) {
        throw std::invalid_argument("avgpool2d expects 4D x");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::avgpool2d(dev, x, kernel, stride, padding);
    }
#endif
    return cpu_ops::avgpool2d(dev, x, kernel, stride, padding);
}

Tensor batchnorm2d(Device& dev, const Tensor& x, const Tensor& gamma, const Tensor& beta,
                   const Tensor& mean, const Tensor& var, float eps) {
    if (x.ndim() != 4) {
        throw std::invalid_argument("batchnorm2d expects 4D x");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::batchnorm2d(dev, x, gamma, beta, mean, var, eps);
    }
#endif
    return cpu_ops::batchnorm2d(dev, x, gamma, beta, mean, var, eps);
}

Tensor softmax(Device& dev, const Tensor& x) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("softmax expects 2D x");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::softmax(dev, x);
    }
#endif
    return cpu_ops::softmax(dev, x);
}

Tensor cross_entropy(Device& dev, const Tensor& logits, const Tensor& target) {
    if (logits.ndim() != 2 || target.ndim() != 1) {
        throw std::invalid_argument("cross_entropy expects 2D logits and 1D target");
    }
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::cross_entropy(dev, logits, target);
    }
#endif
    return cpu_ops::cross_entropy(dev, logits, target);
}

Tensor relu_backward(Device& dev, const Tensor& grad_out, const Tensor& x) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::relu_backward(dev, grad_out, x);
    }
#endif
    return cpu_ops::relu_backward(dev, grad_out, x);
}

Tensor cross_entropy_backward(Device& dev, const Tensor& logits, const Tensor& target) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::cross_entropy_backward(dev, logits, target);
    }
#endif
    return cpu_ops::cross_entropy_backward(dev, logits, target);
}

Tensor transpose(Device& dev, const Tensor& x) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::transpose(dev, x);
    }
#endif
    return cpu_ops::transpose(dev, x);
}

Tensor sum_axis0(Device& dev, const Tensor& x) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::sum_axis0(dev, x);
    }
#endif
    return cpu_ops::sum_axis0(dev, x);
}

void bn_forward_stats(Device& dev, const Tensor& x, Tensor& mean, Tensor& var) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        gl_ops::bn_forward_stats(dev, x, mean, var);
        return;
    }
#endif
    cpu_ops::bn_forward_stats(dev, x, mean, var);
}

void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    Device& dev = param.device();
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        gl_ops::sgd_update(param, grad, lr);
        return;
    }
#endif
    cpu_ops::sgd_update(param, grad, lr);
}

void bn_update_running_stats(Device& dev, Tensor& running_mean, Tensor& running_var,
                             const Tensor& batch_mean, const Tensor& batch_var,
                             float momentum) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        gl_ops::bn_update_running_stats(dev, running_mean, running_var, batch_mean,
                                        batch_var, momentum);
        return;
    }
#endif
    cpu_ops::bn_update_running_stats(dev, running_mean, running_var, batch_mean, batch_var,
                                     momentum);
}

Tensor conv2d_grad_input(Device& dev, const Tensor& grad_out, const Tensor& x,
                         const Tensor& w, int stride, int padding) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::conv2d_grad_input(dev, grad_out, x, w, stride, padding);
    }
#endif
    return cpu_ops::conv2d_grad_input(dev, grad_out, x, w, stride, padding);
}

Tensor conv2d_grad_weight(Device& dev, const Tensor& grad_out, const Tensor& x,
                          const Tensor& w, int stride, int padding) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::conv2d_grad_weight(dev, grad_out, x, w, stride, padding);
    }
#endif
    return cpu_ops::conv2d_grad_weight(dev, grad_out, x, w, stride, padding);
}

Tensor conv2d_grad_bias(Device& dev, const Tensor& grad_out) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::conv2d_grad_bias(dev, grad_out);
    }
#endif
    return cpu_ops::conv2d_grad_bias(dev, grad_out);
}

Tensor col2im(Device& dev, const Tensor& col, int N, int Cin, int H, int W,
              int KH, int KW, int stride, int padding) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::col2im(dev, col, N, Cin, H, W, KH, KW, stride, padding);
    }
#endif
    return cpu_ops::col2im(dev, col, N, Cin, H, W, KH, KW, stride, padding);
}

Tensor grad_out_reshape(Device& dev, const Tensor& grad_out) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::grad_out_reshape(dev, grad_out);
    }
#endif
    return cpu_ops::grad_out_reshape(dev, grad_out);
}

Tensor im2col(Device& dev, const Tensor& x, int KH, int KW, int stride, int padding) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::im2col(dev, x, KH, KW, stride, padding);
    }
#endif
    return cpu_ops::im2col(dev, x, KH, KW, stride, padding);
}

Tensor batchnorm_backward(Device& dev, const Tensor& x, const Tensor& grad_out,
                          const Tensor& gamma, const Tensor& mean, const Tensor& var,
                          float eps, Tensor& grad_gamma, Tensor& grad_beta) {
#ifdef OPENDLL_HAS_OPENGL
    if (dev.info().backend == Backend::OpenGL) {
        return gl_ops::batchnorm_backward(dev, x, grad_out, gamma, mean, var, eps, grad_gamma,
                                          grad_beta);
    }
#endif
    return cpu_ops::batchnorm_backward(dev, x, grad_out, gamma, mean, var, eps, grad_gamma,
                                       grad_beta);
}

}  // namespace opendll
