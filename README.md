# OpenDLL

A C++17 deep learning library (training + inference) built on **OpenGL compute shaders**, with a
backend-agnostic design. Currently ships two backends: **OpenGL** (primary compute) and **CPU**
(numeric reference).

## Features

- **Backend-agnostic Device layer** — `Device / CommandQueue / Buffer / Kernel / Event`; no OpenGL
  types leak above the device layer, so future backends (Vulkan/OpenCL) slot in cleanly.
- **Tensor + Module layer** — `Linear`, `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `AvgPool2d`,
  `Sequential`, plus a preset `ResNet18`.
- **Differentiable ops** — conv2d (im2col + tiled GEMM with fused-transpose backward), linear,
  pooling, batchnorm, softmax, cross-entropy, and their backward passes.
- **Optimizers** — `SGD` and `SGDM` (with momentum), with device-side in-place updates and
  `set_lr` for schedules.
- **state_dict** — save/load weights with PyTorch-aligned keys (`conv1.weight`, `bn1.running_mean`),
  including BatchNorm running stats.
- **Zero third-party dependencies** — WGL context + `glcorearb.h` only; links against
  `opengl32 / user32 / gdi32`.

## Build

Requirements: GCC 13 (MSYS2 ucrt64), CMake 3.16+, OpenGL 4.3+ (Windows).

```bash
cmake -S workspace -B workspace/build -G "MinGW Makefiles"
cmake --build workspace/build
```

Run the tests:

```bash
workspace/build/test_ops.exe           # operator CPU↔OpenGL diff
workspace/build/test_state_dict.exe    # state_dict save/load roundtrip
```

## Quick start

```cpp
using namespace opendll;

auto dev = Device::create(Backend::OpenGL);

ResNet18 net(*dev, 32, 10);            // CIFAR-10 classifier; num_classes=0 → [N,512] backbone features
SGDM sgd(0.01f / 128, 0.9f);           // lr is mean-loss lr divided by batch_size
std::vector<Module*> modules = {&net};

// per batch:
//   Tensor logits = net.forward(x);
//   Tensor grad = cross_entropy_backward(*dev, logits, t);
//   net.backward(grad); sgd.step(modules);

save_state_dict("model.ckpt", net.collect_state());   // weights + running stats
```

See `docs/introduction.md` for the full component reference, and `workspace/examples/` for runnable
training scripts (`train_cifar10.cpp`, `bench_epoch.cpp`, ...).

## Layout

```
workspace/
├── include/opendll/     public headers (device / tensor / ops / module / optimizer / dataset / state_dict / models)
├── src/                 implementation (opengl & cpu backends, ops, models)
├── examples/            training / benchmark scripts
└── tests/               operator diff & state_dict tests
```

## Status

ResNet18 trains end-to-end on CIFAR-10 (10 epochs → ~55% test acc). Next: optimizer-state
checkpointing (resume training) and operator fusion.
