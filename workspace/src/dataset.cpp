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

Dataset load_cifar10(const std::string& images_path, const std::string& labels_path,
                     int channels, int rows, int cols) {
    const std::size_t pixels = static_cast<std::size_t>(channels) * rows * cols;

    std::ifstream fi(images_path, std::ios::binary);
    if (!fi) {
        throw std::runtime_error("cannot open file: " + images_path);
    }
    fi.seekg(0, std::ios::end);
    const std::streamsize size = fi.tellg();
    fi.seekg(0, std::ios::beg);
    const int num_images = static_cast<int>(size) / static_cast<std::streamsize>(pixels * sizeof(float));

    Dataset ds;
    ds.channels = channels;
    ds.rows = rows;
    ds.cols = cols;
    ds.images.resize(static_cast<std::size_t>(num_images) * pixels);
    ds.labels.resize(static_cast<std::size_t>(num_images));
    fi.read(reinterpret_cast<char*>(ds.images.data()),
            static_cast<std::streamsize>(ds.images.size() * sizeof(float)));

    std::ifstream fl(labels_path, std::ios::binary);
    if (!fl) {
        throw std::runtime_error("cannot open file: " + labels_path);
    }
    std::vector<int> labels(static_cast<std::size_t>(num_images));
    fl.read(reinterpret_cast<char*>(labels.data()),
            static_cast<std::streamsize>(num_images * sizeof(int)));

    ds.num_classes = 0;
    for (std::size_t i = 0; i < ds.labels.size(); ++i) {
        ds.labels[i] = labels[i];
        ds.num_classes = std::max(ds.num_classes, labels[i] + 1);
    }
    return ds;
}

}  // namespace opendll
