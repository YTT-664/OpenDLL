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

## M1 — Tensor 层 + matmul / elementwise 算子

> 日期：2026-09-30
> 里程碑：M1

### 目标

在 M0 的 Device 抽象层之上，建立 Tensor 层与首批算子（matmul、relu、add、mul），并用「CPU 参考实现 vs OpenGL 实现」的数值 diff 测试验证正确性。

### 新增结构

```
include/opendll/tensor.hpp    Tensor 层
include/opendll/ops.hpp       算子公开接口
src/tensor.cpp
src/ops/ops.cpp               后端分发
src/ops/cpu_ops.cpp           CPU 参考实现
src/ops/gl_ops.cpp            OpenGL 实现（GLSL + kernel 缓存）
tests/test_ops.cpp            diff 测试
```

### 设计要点

- **Tensor**：contiguous float32，轻量句柄 + `shared_ptr<Buffer>` 共享存储，提供 `upload` / `download`。
- **后端分发**：算子对外统一（`matmul(dev, a, b)`），shape 校验在 `ops.cpp` 统一完成，按 `device.backend` 分发到 `cpu_ops` 或 `gl_ops`。
- **标量参数走 SSBO**：matmul 的 `M/N/K` 经一个参数 buffer（binding 3）传入，而非 uniform，从而完全复用 `BufferBinding` 机制、无需扩展 dispatch 接口。
- **kernel 缓存**：同一 device 的 shader 只编译一次（M2 升级为显式算子注册表）。
- **CPU 参考实现**：经抽象接口 `download` → C++ 循环 → `upload`，作为数值黄金基准。

### 算子清单

| 算子 | 语义 | 实现 |
| --- | --- | --- |
| matmul | C = A @ B，2D | GLSL 8×8 local size，naive 三重循环 |
| relu | y = max(x, 0) | 逐元素 |
| add | c = a + b | 逐元素 |
| mul | c = a * b | 逐元素 |

### diff 测试

`tests/test_ops.cpp` 用固定 seed 的随机数据（[-1,1]），对每个算子分别在 CPU 与 OpenGL 后端计算，比较 `max_abs_diff`：

- matmul：64×32 与 32×48，阈值 `1e-3`
- relu / add / mul：1024 元素，阈值 `1e-4`

### 测试结果

```
[matmul] max_abs_diff=9.53674e-07 PASS
[relu]   max_abs_diff=0          PASS
[add]    max_abs_diff=0          PASS
[mul]    max_abs_diff=0          PASS
ALL OPS PASSED
```

elementwise 精确一致（单次浮点运算无误差），matmul 误差 `9.5e-7` 为 fp32 累加的正常水平。

## M2 — ResNet 算子集（完成）

> 日期：2026-09-30
> 里程碑：M2

### 本期完成：conv2d 与 linear（naive → im2col/GEMM 演进）

按「先 naive 验证正确性，再优化」的策略，conv2d 与 linear 各经历两个阶段：

| 算子 | naive | 优化版 |
| --- | --- | --- |
| conv2d | 三重循环 shader | im2col + 16×16 tiled GEMM |
| linear | 三重循环 shader | 16×16 tiled GEMM |

### 实现要点

- **im2col shader**：`X [N,Cin,H,W]` → 列矩阵 `col [Kcol, Ncol]`（`Kcol=Cin·KH·KW`，`Ncol=N·Hout·Wout`），越界补零。
- **conv GEMM**：`W @ col`，16×16 shared-memory tile，结果直接写回 NCHW 布局（含 bias），省去一次重排。
- **linear GEMM**：`x @ w^T + b`，16×16 tile，`w` 转置访问（B tile 连续读）。
- **标量参数走 SSBO**：与 M1 一致，维度参数经 params buffer 传入，复用 `BufferBinding` 机制。
- **关键修复**：tiled GEMM 需用 `gl_LocalInvocationID` 索引 shared memory（初版误用 global id，M/N>16 时越界导致 diff 失败）。

naive 的 shader 源码（`kConv2dSrc` / `kLinearSrc`）保留在代码中作参考。

### diff 测试结果

```
[conv2d] max_abs_diff=9.53674e-07 PASS
[linear] max_abs_diff=4.29153e-06 PASS
ALL OPS PASSED
```

### 完成：其余 ResNet 算子

```
[maxpool2d]     0          PASS
[avgpool2d]     0          PASS
[batchnorm2d]   1.19209e-07 PASS
[softmax]       2.98023e-08 PASS
[cross_entropy] 0          PASS
```

- **maxpool2d / avgpool2d**：正方形窗口池化；avgpool 对窗口内有效（非 padding）元素取平均。
- **batchnorm2d**：给定 mean/var 的归一化 `y = (x - mean)/sqrt(var + eps) * gamma + beta`。
- **softmax**：沿最后一维，max 减法保证数值稳定。
- **cross_entropy**：融合 log_softmax + nll（logsumexp 形式），返回逐样本损失。

至此 ResNet18/50 所需的 **forward 算子集全部就绪**（共 11 个算子），每个均有 CPU 参考实现 + OpenGL compute shader，统一经 `ops.cpp` 后端分发、随机数据 diff 验证。

### 下一步（M3）

Module 层 + 静态计算图 + backward 梯度，搭 MLP 在 CIFAR 收敛。
