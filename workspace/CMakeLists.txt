cmake_minimum_required(VERSION 3.16)
project(OpenDLL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(opendll
    src/device_factory.cpp
    src/cpu/cpu_device.cpp
    src/tensor.cpp
    src/ops/ops.cpp
    src/ops/cpu_ops.cpp
)
target_include_directories(opendll PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

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
