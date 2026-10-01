#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "opendll/dataset.hpp"
#include "opendll/device.hpp"
#include "opendll/module.hpp"
#include "opendll/ops.hpp"
#include "opendll/optimizer.hpp"

using namespace opendll;

namespace {

// ResNet18 的 BasicBlock：两个 3x3 卷积 + 残差连接。
class BasicBlock final : public Module {
public:
    BasicBlock(Device& dev, int in_planes, int planes, int stride)
        : dev_(dev),
          conv1_(dev, in_planes, planes, 3, stride, 1),
          bn1_(dev, planes),
          relu1_(dev),
          conv2_(dev, planes, planes, 3, 1, 1),
          bn2_(dev, planes),
          relu2_(dev),
          downsample_(stride != 1 || in_planes != planes),
          ds_conv_(dev, in_planes, planes, 1, stride, 0),
          ds_bn_(dev, planes) {}

    Tensor forward(const Tensor& x) override {
        Tensor out = relu1_.forward(bn1_.forward(conv1_.forward(x)));
        out = bn2_.forward(conv2_.forward(out));
        Tensor shortcut = x;
        if (downsample_) {
            shortcut = ds_bn_.forward(ds_conv_.forward(x));
        }
        return relu2_.forward(add(dev_, out, shortcut));
    }

    Tensor backward(const Tensor& grad_out) override {
        Tensor grad = relu2_.backward(grad_out);
        // add 的 identity 双路
        Tensor grad_conv1_path = bn2_.backward(grad);
        grad_conv1_path = conv2_.backward(grad_conv1_path);
        grad_conv1_path = relu1_.backward(grad_conv1_path);
        grad_conv1_path = bn1_.backward(grad_conv1_path);
        Tensor grad_x = conv1_.backward(grad_conv1_path);

        if (downsample_) {
            Tensor grad_shortcut = ds_bn_.backward(grad);
            grad_shortcut = ds_conv_.backward(grad_shortcut);
            grad_x = add(dev_, grad_x, grad_shortcut);
        }
        return grad_x;
    }

    std::vector<std::pair<Tensor*, Tensor*>> parameters() override {
        std::vector<std::pair<Tensor*, Tensor*>> p;
        auto add_all = [&p](Module& m) {
            auto mp = m.parameters();
            p.insert(p.end(), mp.begin(), mp.end());
        };
        add_all(conv1_);
        add_all(bn1_);
        add_all(conv2_);
        add_all(bn2_);
        if (downsample_) {
            add_all(ds_conv_);
            add_all(ds_bn_);
        }
        return p;
    }

private:
    Device& dev_;
    Conv2d conv1_, conv2_;
    BatchNorm2d bn1_, bn2_;
    ReLU relu1_, relu2_;
    bool downsample_;
    Conv2d ds_conv_;
    BatchNorm2d ds_bn_;
};

// 构建 ResNet18（CIFAR10 版）。
struct ResNet {
    Device& dev;
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu;
    std::vector<BasicBlock> layer1, layer2, layer3, layer4;
    Linear fc;

    ResNet(Device& d, int num_classes = 10)
        : dev(d),
          conv1(d, 3, 64, 3, 1, 1),
          bn1(d, 64),
          relu(d),
          fc(d, 512, num_classes) {
        make_layer(layer1, 64, 64, 2, 1);
        make_layer(layer2, 64, 128, 2, 2);
        make_layer(layer3, 128, 256, 2, 2);
        make_layer(layer4, 256, 512, 2, 2);
    }

    void make_layer(std::vector<BasicBlock>& layer, int in_planes, int planes, int blocks,
                    int stride) {
        layer.emplace_back(dev, in_planes, planes, stride);
        for (int i = 1; i < blocks; ++i) {
            layer.emplace_back(dev, planes, planes, 1);
        }
    }

    Tensor forward(const Tensor& x) {
        Tensor out = relu.forward(bn1.forward(conv1.forward(x)));
        for (auto& b : layer1) out = b.forward(out);
        for (auto& b : layer2) out = b.forward(out);
        for (auto& b : layer3) out = b.forward(out);
        for (auto& b : layer4) out = b.forward(out);
        // global avg pool
        const int64_t H = out.dim(2);
        out = avgpool2d(dev, out, static_cast<int>(H), static_cast<int>(H), 0);
        out = out.view({out.dim(0), out.dim(1)});
        return fc.forward(out);
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor grad = fc.backward(grad_out);
        grad = grad.view({grad.dim(0), grad.dim(1), 1, 1});
        for (auto it = layer4.rbegin(); it != layer4.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer3.rbegin(); it != layer3.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer2.rbegin(); it != layer2.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer1.rbegin(); it != layer1.rend(); ++it) grad = it->backward(grad);
        grad = relu.backward(grad);
        grad = bn1.backward(grad);
        return conv1.backward(grad);
    }

    void collect(std::vector<Module*>& modules) {
        modules.push_back(&conv1);
        modules.push_back(&bn1);
        for (auto& b : layer1) modules.push_back(&b);
        for (auto& b : layer2) modules.push_back(&b);
        for (auto& b : layer3) modules.push_back(&b);
        for (auto& b : layer4) modules.push_back(&b);
        modules.push_back(&fc);
    }
};

float eval_accuracy(Device& dev, const Dataset& data, ResNet& net, int batch_size) {
    const int input_dim = data.channels * data.rows * data.cols;
    const int num_classes = data.num_classes;
    const int num = static_cast<int>(data.labels.size());
    int correct = 0;

    for (int start = 0; start + batch_size <= num; start += batch_size) {
        std::vector<float> xb(static_cast<std::size_t>(batch_size) * input_dim);
        std::vector<float> tb(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            const int idx = start + i;
            for (int j = 0; j < input_dim; ++j) {
                xb[static_cast<std::size_t>(i) * input_dim + j] =
                    data.images[static_cast<std::size_t>(idx) * input_dim + j];
            }
            tb[i] = static_cast<float>(data.labels[idx]);
        }
        Tensor x(dev, {batch_size, data.channels, data.rows, data.cols});
        x.upload(xb);
        Tensor logits = net.forward(x);
        std::vector<float> lg;
        logits.download(lg);
        for (int i = 0; i < batch_size; ++i) {
            int best = 0;
            for (int c = 1; c < num_classes; ++c) {
                if (lg[static_cast<std::size_t>(i) * num_classes + c] >
                    lg[static_cast<std::size_t>(i) * num_classes + best]) {
                    best = c;
                }
            }
            if (best == static_cast<int>(tb[i])) ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(num);
}

// 从数据集中选出指定类别，标签重映射到 0..classes.size()-1。
Dataset subset_by_classes(const Dataset& data, const std::vector<int>& classes) {
    Dataset out;
    out.channels = data.channels;
    out.rows = data.rows;
    out.cols = data.cols;
    out.num_classes = static_cast<int>(classes.size());
    const int input_dim = data.channels * data.rows * data.cols;
    for (std::size_t i = 0; i < data.labels.size(); ++i) {
        for (std::size_t c = 0; c < classes.size(); ++c) {
            if (data.labels[i] == classes[c]) {
                const std::size_t base = i * input_dim;
                out.images.insert(out.images.end(), data.images.begin() + base,
                                  data.images.begin() + base + input_dim);
                out.labels.push_back(static_cast<int>(c));
                break;
            }
        }
    }
    return out;
}

}  // namespace

int main() {
    std::cerr << "[dbg] main start\n";
    auto dev = Device::create(Backend::OpenGL);
    std::cerr << "[dbg] device created\n";
    if (!dev) {
        std::cerr << "OpenGL device creation failed\n";
        return 1;
    }
    std::cout << "[device] " << dev->info().name << "\n";

    const std::string data_dir = "D:/OpenDLL/data/CIFAR10/";
    const auto train = load_cifar10(data_dir + "train_images.bin", data_dir + "train_labels.bin",
                                    3, 32, 32);
    const auto test = load_cifar10(data_dir + "test_images.bin", data_dir + "test_labels.bin",
                                   3, 32, 32);

    // 选 3 个子类（0, 1, 2）作为子集
    const std::vector<int> classes = {0, 1, 2};
    const auto train_sub = subset_by_classes(train, classes);
    const auto test_sub = subset_by_classes(test, classes);
    std::cerr << "[dbg] data loaded\n";
    std::cout << "train samples: " << train_sub.labels.size() << "\n";
    std::cout << "test samples: " << test_sub.labels.size() << "\n";
    std::cout << "input dim: " << (train_sub.channels * train_sub.rows * train_sub.cols)
              << "  classes: " << train_sub.num_classes << "\n";

    std::cerr << "[dbg] before ResNet construct\n";
    ResNet net(*dev, train_sub.num_classes);
    std::cerr << "[dbg] ResNet constructed\n";
    std::vector<Module*> modules;
    net.collect(modules);
    std::cout << "total modules: " << modules.size() << "\n";

    const int batch_size = 128;
    const float lr = 0.01f;
    SGD sgd(lr / static_cast<float>(batch_size));

    const int num_train = static_cast<int>(train_sub.labels.size());
    const int input_dim = train_sub.channels * train_sub.rows * train_sub.cols;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < 5; ++epoch) {
        float total_loss = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + batch_size <= num_train; start += batch_size) {
            std::vector<float> xb(static_cast<std::size_t>(batch_size) * input_dim);
            std::vector<float> tb(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                const int idx = start + i;
                for (int j = 0; j < input_dim; ++j) {
                    xb[static_cast<std::size_t>(i) * input_dim + j] =
                        train_sub.images[static_cast<std::size_t>(idx) * input_dim + j];
                }
                tb[i] = static_cast<float>(train_sub.labels[idx]);
            }

            Tensor x(*dev, {batch_size, train_sub.channels, train_sub.rows, train_sub.cols});
            x.upload(xb);
            Tensor t(*dev, {batch_size});
            t.upload(tb);

            Tensor logits = net.forward(x);
            Tensor loss = cross_entropy(*dev, logits, t);
            std::vector<float> lv;
            loss.download(lv);
            float batch_loss = 0.0f;
            for (float v : lv) batch_loss += v;
            total_loss += batch_loss / static_cast<float>(batch_size);
            ++num_batches;

            Tensor grad_logits = cross_entropy_backward(*dev, logits, t);
            net.backward(grad_logits);
            sgd.step(modules);
        }

        const float acc = eval_accuracy(*dev, test_sub, net, batch_size);
        std::cout << "epoch " << epoch << "  loss=" << (total_loss / num_batches)
                  << "  test_acc=" << acc << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "training time: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    return 0;
}
