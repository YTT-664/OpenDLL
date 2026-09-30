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

}  // namespace opendll
