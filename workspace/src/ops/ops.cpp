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

}  // namespace opendll
