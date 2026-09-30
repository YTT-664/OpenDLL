#pragma once

#include <string>
#include <vector>

namespace opendll {

struct Dataset {
    std::vector<float> images;  // [N, rows*cols]，归一化到 [0,1]
    std::vector<int> labels;    // [N]
    int num_classes = 0;
    int rows = 0;
    int cols = 0;
};

// 加载 MNIST / FashionMNIST 的 IDX 格式（big-endian）。
// images_path: *-images-idx3-ubyte，labels_path: *-labels-idx1-ubyte。
Dataset load_mnist(const std::string& images_path, const std::string& labels_path);

}  // namespace opendll
