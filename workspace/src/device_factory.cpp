#include "opendll/device.hpp"

#include "cpu/cpu_device.hpp"

#ifdef _WIN32
#include <windows.h>
// 告知 NVIDIA Optimus 驱动本进程需要高性能 GPU（独显）。
// 定义在 device_factory.cpp（始终被链接），保证符号进入最终 exe 并导出。
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

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
