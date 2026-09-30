# OpenDLL 开发日志

## M0 — Device 抽象层 + OpenGL/CPU 双后端骨架

> 日期：2026-09-30
> 里程碑：M0（对应计划书 `docs/OpenDLL-plan.md` 第 10 节）

### 目标

搭建库的地基：一个后端无关的 Device 抽象层，以及两个可运行的后端——OpenGL（主计算后端）与 CPU（数值参考后端）。验收标准是「内存上传下载、单 kernel dispatch 跑通」。

### 工具链

| 项 | 值 |
| --- | --- |
| 编译器 | GCC 13.2.0（MSYS2 ucrt64） |
| 构建系统 | CMake 3.18.1（MinGW Makefiles） |
| 语言标准 | C++17 |
| 运行时 GPU | Intel Arc，OpenGL 4.6 |
| 第三方依赖 | 无（不依赖 GLFW / GLEW / GLAD） |

### 架构

```
include/opendll/device.hpp   抽象接口
src/device_factory.cpp       Device::create 工厂
src/cpu/                     CPU 参考后端
src/opengl/                  OpenGL 后端（WGL）
tests/smoke_test.cpp         冒烟测试
```

抽象层采用现代 GPU 后端的通用五元组模型，接口语义与具体后端解耦：

- `Device`：工厂 + 资源创建入口（`alloc` / `compile`）
- `CommandQueue`：所有设备操作的唯一入口（`upload` / `download` / `dispatch` / `synchronize`）
- `Buffer`：设备内存块（后端持有真实句柄）
- `Kernel`：已编译的 compute kernel
- `Event`：同步点（本期接口预留）

关键约束：`device.hpp` 中**没有任何 OpenGL 类型泄漏**，`OPENDLL_HAS_OPENGL` 宏控制后端编译，未来接入 Vulkan 只需新增目录并在工厂加一个分支。

### OpenGL 后端实现要点

无 GLFW/GLEW 的前提下，全部用手工方式完成：

1. **上下文创建**（`wgl_context.cpp`）：注册窗口类 → 创建隐藏窗口 → 设置像素格式 → 创建临时 1.1 上下文 → 经 `wglCreateContextAttribsARB` 创建 4.x 核心上下文。
2. **函数指针加载**（`gl_loader.cpp`）：`wglGetProcAddress` 加载 1.2+ 函数，1.1 函数回退到 `opengl32.dll`；集中到 `Funcs` 结构体避免与 `glcorearb.h` 的 extern 声明冲突。
3. **Buffer = SSBO**：`glGenBuffers` + `glBufferData`，上传/下载走 `glMapBufferRange`。
4. **Kernel = program**：`glCreateShader(GL_COMPUTE_SHADER)` → 编译 → 链接，失败时抛带日志的异常。
5. **dispatch**：`glBindBufferBase` 绑定到 shader 的 binding point → `glUseProgram` → `glDispatchCompute` → `glMemoryBarrier`。

### CPU 后端

`CpuBuffer`（`std::vector<std::byte>`）+ `CpuQueue` 直连 `memcpy`。`dispatch` 在 M0 为占位实现——真实算子的 CPU 参考执行留待 M1，用于与 OpenGL 后端做数值 diff 验证。

### 冒烟测试内容

`tests/smoke_test.cpp` 覆盖两条路径：

1. **CPU 后端往返**：`alloc` 4 个 float → `upload` → `download`，校验数据一致。
2. **OpenGL 后端 compute 执行**：编译一个 trivial compute shader（对 buffer 中每个 float `+1`）→ 上传 8 个 float → `dispatch` → `synchronize` → `download`，校验每个元素 `+1`。

### 测试结果

```
[CPU] upload/download roundtrip: PASS
[OpenGL] OpenGL 4.6.0 / Intel(R) Arc(TM) Graphics
[OpenGL] max workgroup size=1024 shared mem=32768
[OpenGL] compute dispatch (+1): PASS
ALL TESTS PASSED
```

整条 compute 管线（SSBO 上传 → shader 编译链接 → `glDispatchCompute` → 数据回读）在真实 GPU 上验证通过。

### 下一步（M1）

Tensor 层 + matmul + elementwise 算子，并建立「CPU 参考实现 vs OpenGL 实现」的数值 diff 测试框架。
