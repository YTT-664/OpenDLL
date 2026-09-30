#include "opendll/device.hpp"

#include "cpu/cpu_device.hpp"

#ifdef OPENDLL_HAS_OPENGL
#include "opengl/opengl_device.hpp"
#endif

namespace opendll {

std::unique_ptr<Device> Device::create(Backend b) {
    switch (b) {
        case Backend::CPU:
            return std::make_unique<CpuDevice>();
        case Backend::OpenGL:
#ifdef OPENDLL_HAS_OPENGL
            return OpenGLDevice::create();
#else
            return nullptr;
#endif
    }
    return nullptr;
}

}  // namespace opendll
