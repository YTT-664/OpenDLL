#pragma once

#include <string>
#include <vector>

namespace opendll {

struct Dataset {
    std::vector<float> images;  // [N, channels*rows*cols]，归一化到 [0,1]
    std::vector<int> labels;    // [N]
    int num_classes = 0;
    int channels = 1;
    int rows = 0;
    int cols = 0;
};

// 加载 MNIST / FashionMNIST 的 IDX 格式（big-endian）。
// images_path: *-images-idx3-ubyte，labels_path: *-labels-idx1-ubyte。
Dataset load_mnist(const std::string& images_path, const std::string& labels_path);

// 加载 CIFAR10 的 flat 二进制（float32 images，NCHW 布局；int32 labels）。
// 样本数由文件大小自动推断。
Dataset load_cifar10(const std::string& images_path, const std::string& labels_path,
                     int channels, int rows, int cols);

}  // namespace opendll
