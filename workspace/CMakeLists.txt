cmake_minimum_required(VERSION 3.16)
project(OpenDLL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 是否把 OpenGL 后端路由到 NVIDIA 独显（Optimus）。默认 ON；关闭后回落集显。
option(OPENDLL_ENABLE_NVOPTIMUS "Route OpenGL backend to NVIDIA discrete GPU" ON)

add_library(opendll
    src/device_factory.cpp
    src/cpu/cpu_device.cpp
    src/tensor.cpp
    src/ops/ops.cpp
    src/ops/cpu_ops.cpp
    src/module.cpp
    src/optimizer.cpp
    src/dataset.cpp
)
target_include_directories(opendll PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(OPENDLL_ENABLE_NVOPTIMUS AND WIN32)
    target_compile_definitions(opendll PUBLIC OPENDLL_ENABLE_NVOPTIMUS)
endif()

# OpenGL 后端（WGL）仅在 Windows 下编译
if(WIN32)
    target_sources(opendll PRIVATE
        src/opengl/gl_loader.cpp
        src/opengl/wgl_context.cpp
        src/opengl/opengl_device.cpp
        src/ops/gl_ops.cpp
    )
    target_compile_definitions(opendll PUBLIC OPENDLL_HAS_OPENGL)
    target_link_libraries(opendll PRIVATE opengl32 user32 gdi32)
endif()

# 测试
enable_testing()
add_executable(smoke_test tests/smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE opendll)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE opendll)

add_test(NAME smoke_test COMMAND smoke_test)
add_test(NAME test_ops COMMAND test_ops)

# 训练示例
add_executable(train_mlp examples/train_mlp.cpp)
target_link_libraries(train_mlp PRIVATE opendll)

add_executable(train_resnet examples/train_resnet.cpp)
target_link_libraries(train_resnet PRIVATE opendll)

add_executable(profile_resnet examples/profile_resnet.cpp)
target_link_libraries(profile_resnet PRIVATE opendll)

# GPU 独显测试（WGL_NV_gpu_affinity，仅 Windows）
if(WIN32)
    add_executable(test_gpu examples/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE opengl32 user32 gdi32)
endif()
