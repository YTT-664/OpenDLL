#include "opendll/dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>

namespace opendll {

namespace {

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open file: " + path);
    }
    f.seekg(0, std::ios::end);
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<unsigned char> buf(static_cast<std::size_t>(size));
    if (!buf.empty()) {
        f.read(reinterpret_cast<char*>(buf.data()), size);
    }
    return buf;
}

int read_be32(const std::vector<unsigned char>& buf, std::size_t offset) {
    return (static_cast<int>(buf[offset]) << 24) |
           (static_cast<int>(buf[offset + 1]) << 16) |
           (static_cast<int>(buf[offset + 2]) << 8) | static_cast<int>(buf[offset + 3]);
}

}  // namespace

Dataset load_mnist(const std::string& images_path, const std::string& labels_path) {
    const auto img = read_file(images_path);
    const auto lab = read_file(labels_path);

    const int num_images = read_be32(img, 4);
    const int rows = read_be32(img, 8);
    const int cols = read_be32(img, 12);
    const int num_labels = read_be32(lab, 4);

    if (num_images != num_labels) {
        throw std::runtime_error("images/labels count mismatch");
    }

    Dataset ds;
    ds.rows = rows;
    ds.cols = cols;
    const std::size_t pixels = static_cast<std::size_t>(rows) * cols;
    ds.images.resize(static_cast<std::size_t>(num_images) * pixels);
    ds.labels.resize(static_cast<std::size_t>(num_labels));

    for (std::size_t i = 0; i < ds.images.size(); ++i) {
        ds.images[i] = static_cast<float>(img[16 + i]) / 255.0f;
    }
    ds.num_classes = 0;
    for (std::size_t i = 0; i < ds.labels.size(); ++i) {
        ds.labels[i] = static_cast<int>(lab[8 + i]);
        ds.num_classes = std::max(ds.num_classes, ds.labels[i] + 1);
    }
    return ds;
}

}  // namespace opendll
