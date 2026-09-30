# OpenDLL 开发计划书

> 版本：v0.1
> 日期：2026-09-30
> 目标平台：C++17 / OpenGL 4.3+ compute shader（可替换后端）
> 短期目标：在框架内复现并训练 ResNet18 / ResNet50

---

## 1. 项目概述

### 1.1 目标

构建一个基于 compute shader 管线的 C++17 深度学习训练与推理库。第一阶段采用「静态计算图 + 手动 backward」的早期 torch 模式，在此基础上逐步演进为基于节点的 autograd 自动求导。

### 1.2 短期里程碑

在框架内完整复现 ResNet18 与 ResNet50，支持：

- 前向推理（forward）
- 反向训练（backward + 优化器）
- 在 CIFAR-10 / CIFAR-100 上验证收敛正确性，最终支持 ImageNet 规模
- 训练 / 推理两种模式（影响 BatchNorm 等算子的行为）

### 1.3 非目标（本期明确不做）

- 分布式 / 多卡训练
- 半精度（fp16）与混合精度训练
- 动态图（eager mode）
- ONNX / TorchScript 等模型交换格式
- 算子自动融合与全量 kernel 性能优化（仅在正确性达标后按需做热点优化）

---

## 2. 技术选型与约束

| 项 | 选择 | 说明 |
| --- | --- | --- |
| 语言 | C++17 | `if constexpr`、`std::variant`、`std::optional`、结构化绑定 |
| 主后端 | OpenGL 4.3+ compute shader | GLSL 430，SSBO `std430` 布局 |
| 后端抽象 | 自定义 Device 抽象层 | 后端不锁死 OpenGL，未来可加 Vulkan / CPU |
| 参考后端 | CPU 单线程/OpenMP 参考实现 | 用于数值验证与 diff 测试 |
| 数据布局 | NCHW | 与主流框架一致，利于 conv / bn 实现 |
| 数值精度 | fp32 | GLSL `float` 即 32-bit |
| 构建系统 | CMake | 跨平台 |

---

## 3. 总体架构

### 3.1 分层

```
┌────────────────────────────────────────────────────────────┐
│  Module 层                                                  │
│  Conv2d / BatchNorm2d / Linear / ReLU / MaxPool / ...       │
│  组合可微算子，持有参数与 running stats                     │
├────────────────────────────────────────────────────────────┤
│  可微算子层（autograd 的核心抽象）                          │
│  matmul / conv2d / bn / relu / pool / add / softmax ...     │
│  每个算子：forward + 梯度规则（后端无关，只写一次）          │
├────────────────────────────────────────────────────────────┤
│  Tensor 层                                                  │
│  shape / stride / dtype / device / grad 缓冲 / 图节点        │
├────────────────────────────────────────────────────────────┤
│  Device 抽象层                                              │
│  Device / CommandQueue / Buffer / Kernel / Event             │
│  OpenGL 实现（本期） · CPU 参考（测试） · Vulkan（预留）      │
└────────────────────────────────────────────────────────────┘
```

### 3.2 职责边界

- **算子层是唯一知道「梯度如何计算」的层**。Module 只做算子组合，不写梯度公式。
- **Device 抽象层是唯一接触后端 API 的层**。Tensor 与算子层不得出现任何 `GLuint` / GLSL 源码之外的 OpenGL 类型。
- 每个算子的**语义与梯度规则只定义一次**（后端无关），各后端只负责「如何计算」。
- 每个算子的**数值正确性**由 CPU 参考后端统一验证，GPU 后端不必单独写测试。

---

## 4. Device 抽象层（核心设计）

### 4.1 设计原则

采用现代 GPU 后端的通用五元组模型：**设备 + 命令队列 + 内存 + 内核 + 事件**。这一模型同时覆盖 OpenGL、Vulkan、CUDA、OpenCL 的能力，保证抽象层的通用性。

1. 所有后端操作通过 `CommandQueue` 提交，支持**异步**；OpenGL 后端内部可用 `glFinish`/fence 降级实现，但接口语义必须异步，为未来 Vulkan 铺路。
2. 内存上传/下载显式化，算子层与主机数据的交互只发生在「边界」。
3. kernel 以字符串源码形式编译并缓存，避免重复编译。

### 4.2 核心抽象接口（示意）

```cpp
enum class Backend { OpenGL, Vulkan, CPU };

// 后端句柄类型，内部隐藏，对外仅暴露轻量包装
class Buffer;   // 设备内存块
class Kernel;   // 已编译的 compute kernel
class Event;    // 同步点（fence）

// 命令队列：唯一的命令提交入口
class CommandQueue {
public:
    // 分派 kernel，返回可等待的事件
    Event dispatch(const Kernel& k,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   std::span<const BufferBinding> bindings);

    // 主机 <-> 设备拷贝（异步入队）
    void upload(const Buffer& dst, const void* src, size_t bytes);
    void download(const void* src, void* dst, size_t bytes);   // src 为 Buffer
    void wait(const Event& e);   // 显式声明跨 kernel 依赖
    void synchronize();          // host 阻塞至队列清空
};

// 顶层设备：创建资源与后端对象
class Device {
public:
    static std::unique_ptr<Device> create(Backend b);

    Buffer  alloc(size_t bytes);
    Kernel  compile(std::string_view source, std::string_view entry);
    CommandQueue& main_queue();
    void synchronize();

    struct Info {
        Backend backend;
        std::string name;
        uint32_t max_workgroup_size;
        uint32_t max_shared_memory;   // OpenGL 通常 16KB
        bool supports_atomic_float;
    };
    Info info() const;
};
```

### 4.3 后端实现映射

| 抽象 | OpenGL | CPU 参考 | Vulkan（预留） |
| --- | --- | --- | --- |
| `Buffer` | SSBO | `std::vector<std::byte>` | `VkBuffer` |
| `Kernel` | linked `GLuint program` | 函数指针/闭包 | `VkPipeline` |
| `CommandQueue` | GL 命令流 | 直接执行/OpenMP 循环 | `VkQueue` + `VkCommandBuffer` |
| `Event` | `glFenceSync` | 空操作 | `VkFence`/`VkSemaphore` |
| `dispatch` | `glDispatchCompute` | 并行 for | `vkCmdDispatch` |
| `synchronize` | `glFinish` | 空操作 | `vkQueueWaitIdle` |

### 4.4 同步模型

- 同一 `CommandQueue` 内命令天然串行；跨队列/跨流用 `Event` 表达依赖。
- 算子层在「一个 op 的输出被下一个 op 读取」时，由运行时统一插入 `wait` 或依赖 barrier（OpenGL 后端对应 `glMemoryBarrier`）。
- 主机只有在 `download` 或取回数据时才 `synchronize`，训练主循环中梯度回传下载频率低，避免每 op 一次 `glFinish`。

### 4.5 kernel 注册与复用

算子层通过「kernel 注册表」获得后端实现，而非硬编码后端调用：

```
op 定义（name + 超参 + 输入输出描述 + 梯度规则）
      │
      ▼
后端注册表  OpenGL: conv2d_fwd.glsl / conv2d_bwd.glsl
            CPU:    conv2d_fwd.hpp (C++ 循环)
            Vulkan: conv2d_fwd.spv（预留）
```

这样新增后端只需补充 kernel 源码，算子语义层零改动。

---

## 5. Tensor 与内存管理

### 5.1 Tensor 结构

```cpp
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;   // 本期可先只支持 contiguous
    DataType dtype;                // 本期仅 float32
    std::shared_ptr<Storage> data; // 设备内存 + 偏移（支持 view）
    std::shared_ptr<Storage> grad; // 梯度，惰性分配
    bool requires_grad;
};
```

- 本期先限定 **contiguous NCHW**，`stride` 字段预留但不实现任意 stride，降低复杂度。
- 参数（weight/bias）与中间激活共用同一 `Tensor` 类型，靠 `requires_grad` 区分。

### 5.2 内存布局与对齐

- NCHW 布局；`std430` 下数组元素按 16 字节对齐，buffer 内部以标量 `float[]` 存储，避免 `vec3` 导致的 padding 陷阱。
- 权重、梯度、激活分属不同 buffer 池，减少碎片与重复分配。

### 5.3 内存池

- 后端的 `alloc` 由池化分配器承接，按 size 分桶复用。
- 训练中每步产生的中间激活 buffer 频繁申请释放，池化可显著降低 `glBufferData` 次数。

---

## 6. 算子层

### 6.1 算子抽象

```cpp
struct OpNode {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;

    virtual void forward() = 0;
    virtual void backward() = 0;   // 依据上游传来的输出梯度，累加输入梯度
};
```

梯度累加语义必须明确：`backward` 将 `grad_output` 沿链式法则**累加**到各输入张量的 `grad` 中，因为同一张量可能被多条路径复用（残差连接中的 `add`）。

### 6.2 梯度规则（后端无关，只写一次）

梯度公式与 shape 推导放在算子语义层，backward 的实现有两种方式，按阶段选择：

- **第一阶段**：backward 直接实现为专用 kernel（正确性优先，性能次之）。
- **第二阶段**：热点算子（conv）的 backward 用「转置卷积 / gemm」等 primitive 组合实现，减少专用 kernel 数量。

### 6.3 ResNet 所需算子清单

以下为复现 ResNet18/50 的**完整且充分**的算子集合：

| # | 算子 | forward | backward |
| --- | --- | --- | --- |
| 1 | conv2d（含 bias，支持 stride/padding，dilation 可选） | 是 | 对 input / weight / bias |
| 2 | batchnorm2d（训练/推理双模式） | 是 | 对 input / gamma / beta |
| 3 | relu | 是 | 是 |
| 4 | maxpool2d（记录 argmax 索引） | 是 | 按索引回传 |
| 5 | avgpool2d（含全局 adaptive → 1x1） | 是 | 均分回传 |
| 6 | linear（matmul + bias） | 是 | 对 input / weight / bias |
| 7 | add（残差连接） | 是 | identity 双路传递 |
| 8 | cross_entropy（融合 softmax + nll，数值稳定） | 是 | 是 |
| 9 | 权重初始化（He/Kaiming 正态） | 主机生成后上传 | — |

> 说明：ResNet18（BasicBlock）与 ResNet50（Bottleneck）算子需求完全一致，差异仅在结构配置，故一套算子即可覆盖两者。

数值稳定性要点：

- `cross_entropy` 直接实现为 `log_softmax + nll` 融合 kernel，避免单独 softmax 的溢出。
- `avgpool` / `bn` 的均值方差计算注意 fp32 累积误差，必要时用两趟（先求和再求方差）。

---

## 7. Module 与计算图

### 7.1 Module 接口

```cpp
class Module {
public:
    virtual Tensor forward(const Tensor& x) = 0;   // 组合算子
    std::vector<Tensor*> parameters();             // 收集需更新参数
    void train(bool on);                            // 切换 train/eval
    void zero_grad();
};
```

### 7.2 静态计算图（第一阶段）

- `forward` 内部按固定顺序调用算子，构建一个**静态 `OpNode` 列表**（每张图实例独立，或每次 forward 重建）。
- 反向传播：对图节点**逆拓扑序遍历**，依次调用 `backward`。
- 残差结构中 `add` 的输入被两条路径共享，逆序回传时依赖「梯度累加」语义（见 6.1），需在运行时保证每个节点的 `backward` 恰执行一次、梯度正确累加。

### 7.3 向 autograd 演进（第二阶段）

- 得益于算子层已按 `OpNode` 抽象，演进时只需在 `Tensor` 上记录「生产它的 OpNode + 输入引用」，构成真正的计算图。
- `forward` 自动记录，`backward` 从标量损失出发做拓扑反向遍历，Module 层零改动。
- 静态图阶段积累的算子梯度规则、shape 推导、数值测试可直接复用。

---

## 8. 训练基础设施

### 8.1 数据加载

- `Dataset` / `DataLoader` 接口；CIFAR 与 ImageNet 各一实现。
- 预处理（resize、normalize、随机裁剪/翻转）在主机端完成，异步上传到设备。

### 8.2 优化器

- 优先实现 **SGD + momentum + weight_decay**（ResNet 标准配置，含对 BN 参数不做 weight decay 的惯例）。
- 学习率调度：step decay 或 cosine；随训练规模逐步加入。

### 8.3 初始化

- 卷积/全连接用 **Kaiming（He）正态初始化**，BN 的 `gamma=1, beta=0`。
- 初始化对 ResNet 收敛至关重要，需在参考实现里固化并与 torch 对齐。

### 8.4 损失与精度

- 损失：`cross_entropy`（8 号算子）。
- 精度报告：top-1 / top-5 准确率。

---

## 9. 测试与验证

### 9.1 CPU 参考后端

- 每个算子提供纯 C++ 语义等价实现，作为**唯一正确性基准**。
- 任意随机输入下，GPU 后端输出与 CPU 参考的 `max_abs_diff < tol`（fp32 建议 `1e-4` 量级，按算子复杂度调整）。

### 9.2 数值梯度检查

- 对每个算子的 backward，用中心差分 `(f(x+h)-f(x-h))/(2h)` 校验解析梯度。
- 覆盖 conv2d、batchnorm、linear、pool、cross_entropy，这是反向实现正确性的黄金标准。

### 9.3 端到端收敛验证

- 先在 CIFAR-10 上训练 ResNet18，验证 loss 下降与准确率与参考实现（torch 同配置）**同量级**。
- 再扩展 CIFAR-100 与 ResNet50，最终 ImageNet。

---

## 10. 里程碑路线图

| 阶段 | 内容 | 验收标准 |
| --- | --- | --- |
| M0 | Device 抽象层 + OpenGL 后端 + CPU 参考后端 | 内存上传下载、单 kernel dispatch 跑通 |
| M1 | Tensor + matmul + elementwise 算子 | 与 CPU 参考 diff 通过 |
| M2 | 完整 ResNet 算子集（conv/bn/pool/linear/add/ce） | 每个算子前向+反向数值梯度通过 |
| M3 | Module 层 + 静态计算图 + 手动 backward | 手动搭 MLP 在 CIFAR 收敛 |
| M4 | 复现 ResNet18（训练+推理） | CIFAR-10 准确率对齐参考 |
| M5 | ResNet50 + 优化（tiled matmul / im2col conv） | CIFAR-100 收敛，性能可接受 |
| M6 | autograd 节点图演进 | 移除手动 backward，Module 零改动切换 |
| M7（可选） | ImageNet 规模 + 热点算子融合 | 端到端训练吞吐达标 |

---

## 11. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| OpenGL compute 性能上限 | 训练吞吐低 | M0 即 benchmark naive matmul；必要时走 fused kernel 或预留 Vulkan |
| 跨 kernel 同步开销 | 多小算子串行慢 | 算子融合、内存池、异步队列 |
| 数值误差累积 | 梯度不收敛 | CPU 参考 diff + 数值梯度检查，关键算子两趟计算 |
| GLSL 调试困难 | 排查效率低 | 以 CPU 参考为主调试，GPU kernel 保持简单 |
| BatchNorm train/eval 行为混淆 | 准确率异常 | Module 强制显式 `train(bool)`，测试期冻结 running stats |
| 初始化不当 | 不收敛 | 与 torch 对齐 He 初始化，固化到参考测试 |

---

## 附录 A：ResNet18 / ResNet50 结构参数

| 组件 | ResNet18 | ResNet50 |
| --- | --- | --- |
| 首层 | 7x7 conv, stride 2, 64ch + BN + ReLU + maxpool | 同左 |
| Block 类型 | BasicBlock（3x3, 3x3） | Bottleneck（1x1, 3x3, 1x1） |
| stage 层数 | [2,2,2,2] | [3,4,6,3] |
| stage 通道 | [64,128,256,512] | [256,512,1024,2048] |
| 下采样 | stride=2 时 1x1 conv shortcut | 同左 |
| 尾部 | global avg pool → fc(1000) | 同左 |
