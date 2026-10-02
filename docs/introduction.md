# OpenDLL 简介

OpenDLL 是一个基于 **OpenGL compute shader** 的 C++17 深度学习库（训练 + 推理），
设计上后端无关——当前提供 **OpenGL**（主计算后端）与 **CPU**（数值参考后端）两个实现。

本文说明「上层暴露组件」——也就是编写训练/推理脚本时直接接触的 API：
**Tensor、Module 体系、Optimizer、Dataset、StateDict**，并简要提及 Device 与算子层。

## 分层结构

```
┌─ 应用层   examples/train_resnet.cpp 等训练脚本
├─ 上层组件（本文） ──────────────────────────────
│   Tensor     数据容器（shape / 上传下载 / view）
│   Module 层  网络层（Linear / Conv2d / BatchNorm2d / ReLU / Sequential）
│   Optimizer  参数更新（SGD / SGDM）
│   Dataset    数据加载（MNIST / FashionMNIST / CIFAR-10）
│   StateDict  权重存取（save_state_dict / load_state_dict）
├─ 算子层 ops   可微算子（matmul / conv2d / ... 及其 backward）
└─ 设备层 device  Device / CommandQueue / Buffer / Kernel 后端抽象
```

## Tensor —— 数据容器

`include/opendll/tensor.hpp`

Tensor 是 contiguous float32 的轻量句柄，底层存储由 `shared_ptr<Buffer>` 共享，
因此**拷贝 Tensor 只是共享同一块设备内存**（零拷贝），符合深度学习框架里「张量是句柄」的语义。

```cpp
Tensor x(dev, {128, 3, 32, 32});     // 构造：指定设备与 shape
x.upload(data);                       // 主机 → 设备（float* 或 std::vector<float>）
std::vector<float> out;
x.download(out);                      // 设备 → 主机

x.shape();  x.dim(0);  x.ndim();  x.numel();   // 形状信息
Tensor y = x.view({128, 3 * 32 * 32});         // 改 shape，共享同一块 buffer
```

## Module 层 —— 可组合网络层

`include/opendll/module.hpp`

所有网络层继承自 `Module` 基类，统一三个接口：

```cpp
class Module {
    virtual Tensor forward(const Tensor& x) = 0;      // 前向
    virtual Tensor backward(const Tensor& grad_out) = 0;  // 反向（返回对输入的梯度）
    virtual std::vector<std::pair<Tensor*, Tensor*>> parameters() = 0;  // (参数, 梯度) 对
};
```

`forward` 组合算子、`backward` 回传梯度（静态计算图、手动反向）、`parameters` 供优化器收集。
内置层：

| 层 | 构造 | 语义 |
| --- | --- | --- |
| `Linear` | `(dev, in_features, out_features)` | `y = x @ w^T + b`，Xavier 初始化 |
| `Conv2d` | `(dev, in_c, out_c, kernel, stride, padding)` | 2D 卷积 + bias，He 初始化 |
| `BatchNorm2d` | `(dev, num_features, eps, momentum)` | 批归一化，`train(bool)` 切换 batch/running 统计 |
| `ReLU` | `(dev)` | 逐元素激活 |
| `Sequential` | 无参，`add(shared_ptr<Module>)` | 顺序 forward / 逆序 backward |

**自定义层**：继承 `Module` 实现三个接口即可。残差块 `BasicBlock` 就是这么做出来的——
`forward` 里把 `Conv2d / BatchNorm2d / ReLU` 串起来再 `add` 残差，`backward` 里逆序回传、短路支路梯度相加。

**预置模型**：`ResNet18`（`include/opendll/models/resnet.hpp`）已入库，构造
`ResNet18(dev, in_size, num_classes)`。`num_classes=0` 时不生成分类头，`forward` 直接输出
global avg pool 之后的 `[N, 512]` 特征向量（骨干网络尾部）。

## Optimizer —— 参数更新

`include/opendll/optimizer.hpp`

提供两个优化器，更新都在**设备端就地完成**（不逐 batch 下载/上传参数）：

| 优化器 | 更新规则 | 说明 |
| --- | --- | --- |
| `SGD` | `param -= lr * grad` | 朴素 SGD，无 momentum |
| `SGDM` | `v = momentum*v + grad; param -= lr*v` | SGD with momentum，惰性为每个参数维护 velocity |

```cpp
SGDM sgd(lr_mean / batch_size, 0.9f);   // 见下方 lr 语义说明
std::vector<Module*> modules;
// ... 把网络各层塞进 modules ...
sgd.step(modules);                       // 对每个 (param, grad) 就地更新
sgd.set_lr(new_lr);                      // 训练中可动态调 lr（配合 lr schedule）
```

**lr 语义**：`cross_entropy` 返回逐样本 loss、`cross_entropy_backward` 返回**未平均**的梯度，
所以传入优化器的 `lr` 应为「mean-loss 语义的学习率 ÷ batch_size」。例如希望 mean-loss 下
`lr=0.01`，则 `SGDM(0.01f / 128, 0.9f)`。注意 momentum 会把稳态有效学习率放大到
`lr_mean / (1 - momentum)`，因此无 weight decay 时建议 `lr_mean ≤ 1e-2`，并用 `set_lr`
配合 step-decay 压住后期。

## Dataset —— 数据加载

`include/opendll/dataset.hpp`

```cpp
struct Dataset {
    std::vector<float> images;  // [N, channels*rows*cols]，归一化到 [0,1]
    std::vector<int>   labels;  // [N]
    int num_classes, channels, rows, cols;
};

Dataset train = load_mnist(images_path, labels_path);              // IDX 格式
Dataset train = load_cifar10(img_path, lab_path, 3, 32, 32);       // flat float32 二进制
```

## 存储与读取 —— state_dict

`include/opendll/state_dict.hpp`

模型权重通过 **state_dict**（`name → 张量` 的映射）保存与加载。命名采用**手动命名**（模块树里
显式 `set_name`，叶子层自动追加 `.weight` / `.bias` 等后缀），key 与 PyTorch 对齐——如
`conv1.weight`、`layer1.0.conv1.weight`、`bn1.running_mean`。

每个模块通过 `collect_state()` 返回 `StateEntry{name, tensor, is_buffer}` 列表；其中
`is_buffer` 标记**非训练状态**（如 BatchNorm 的 running stats），它们参与保存、但不参与梯度更新。

```cpp
ResNet18 net(*dev, 32, 10);

// 保存：把全部可训练参数 + running stats 序列化到文件
save_state_dict("model.ckpt", net.collect_state());

// 读取：按 name 匹配填充到已有模型（缺项或 numel 不匹配会抛 std::runtime_error）
ResNet18 loaded(*dev, 32, 10);
load_state_dict("model.ckpt", loaded.collect_state());
```

格式为自定义二进制（magic `ODLL` + 逐条 `name_len/name/numel/float32[]`）。注意加载前需先按
**相同结构**实例化模型——`load_state_dict` 只填充权重、不重建结构（结构由代码定义、权重由文件提供）。

## Device —— 后端抽象（简要）

`include/opendll/device.hpp`

通过工厂创建后端，抽象层不泄漏任何 OpenGL 类型：

```cpp
auto dev = Device::create(Backend::OpenGL);   // 或 Backend::CPU
dev->alloc(bytes);        // 分配 Buffer
dev->compile(glsl);       // 编译 Kernel
dev->main_queue();        // CommandQueue：upload / download / dispatch / synchronize
```

## 最小训练循环

```cpp
using namespace opendll;

auto dev = Device::create(Backend::OpenGL);
auto train = load_cifar10(".../train_images.bin", ".../train_labels.bin", 3, 32, 32);

ResNet18 net(*dev, 32, 10);           // 预置模型
std::vector<Module*> modules = {&net};
SGD sgd(0.01f / 128);

for (int epoch = 0; epoch < 5; ++epoch) {
    for (int start = 0; start + 128 <= N; start += 128) {
        Tensor x(*dev, {128, 3, 32, 32}), t(*dev, {128});
        x.upload(/* batch 数据 */);
        t.upload(/* batch 标签 */);

        Tensor logits = net.forward(x);
        Tensor loss   = cross_entropy(*dev, logits, t);       // 算子层
        Tensor grad   = cross_entropy_backward(*dev, logits, t);
        net.backward(grad);
        sgd.step(modules);
    }
}
```

完整可运行的例子见 `workspace/examples/`（`train_mlp.cpp`、`train_resnet.cpp`、
`train_cifar10.cpp`、`bench_epoch.cpp`、`profile_resnet.cpp`）。
