#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "opendll/dataset.hpp"
#include "opendll/device.hpp"
#include "opendll/module.hpp"
#include "opendll/ops.hpp"
#include "opendll/optimizer.hpp"

using namespace opendll;

namespace {

// ResNet18 的 BasicBlock（含 bn 的 train/eval 切换）。
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

    void train_mode(bool on) {
        bn1_.train(on);
        bn2_.train(on);
        if (downsample_) ds_bn_.train(on);
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

struct ResNet {
    Device& dev;
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu;
    std::vector<BasicBlock> layer1, layer2, layer3, layer4;
    Linear fc;
    int64_t last_H_ = 0, last_W_ = 0;  // layer4 输出的空间尺寸（供 avgpool backward 用）

    ResNet(Device& d, int num_classes = 10)
        : dev(d), conv1(d, 3, 64, 3, 1, 1), bn1(d, 64), relu(d), fc(d, 512, num_classes) {
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
        last_H_ = out.dim(2);
        last_W_ = out.dim(3);
        out = avgpool2d(dev, out, static_cast<int>(last_H_), static_cast<int>(last_H_), 0);
        out = out.view({out.dim(0), out.dim(1)});
        return fc.forward(out);
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor grad = fc.backward(grad_out);
        grad = grad.view({grad.dim(0), grad.dim(1), 1, 1});
        // global avg pool 的 backward：把 [N,C,1,1] 广播回 [N,C,H,W]
        grad = avgpool2d_backward(dev, grad, static_cast<int>(last_H_), static_cast<int>(last_W_));
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

    void train_mode(bool on) {
        bn1.train(on);
        for (auto& b : layer1) b.train_mode(on);
        for (auto& b : layer2) b.train_mode(on);
        for (auto& b : layer3) b.train_mode(on);
        for (auto& b : layer4) b.train_mode(on);
    }
};

// 合并两个数据集，固定 seed 打乱，按 6:2:2 划分 train/val/test。
void split_622(const Dataset& a, const Dataset& b, unsigned seed, Dataset& train,
               Dataset& val, Dataset& test) {
    const int input_dim = a.channels * a.rows * a.cols;
    const int total = static_cast<int>(a.labels.size() + b.labels.size());

    std::vector<float> all_images;
    all_images.reserve(a.images.size() + b.images.size());
    all_images.insert(all_images.end(), a.images.begin(), a.images.end());
    all_images.insert(all_images.end(), b.images.begin(), b.images.end());
    std::vector<int> all_labels = a.labels;
    all_labels.insert(all_labels.end(), b.labels.begin(), b.labels.end());

    std::vector<int> idx(total);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    const int n_train = total * 6 / 10;
    const int n_val = total * 2 / 10;

    auto pick = [&](int lo, int hi, Dataset& d) {
        d.channels = a.channels;
        d.rows = a.rows;
        d.cols = a.cols;
        d.num_classes = a.num_classes;
        for (int k = lo; k < hi; ++k) {
            const int src = idx[k];
            d.images.insert(d.images.end(), all_images.begin() + src * input_dim,
                            all_images.begin() + (src + 1) * input_dim);
            d.labels.push_back(all_labels[src]);
        }
    };

    pick(0, n_train, train);
    pick(n_train, n_train + n_val, val);
    pick(n_train + n_val, total, test);
}

float evaluate(Device& dev, const Dataset& data, ResNet& net, int batch_size) {
    const int input_dim = data.channels * data.rows * data.cols;
    const int num_classes = data.num_classes;
    const int num = static_cast<int>(data.labels.size());
    int correct = 0;
    int total = 0;

    for (int start = 0; start < num; start += batch_size) {
        const int b = std::min(batch_size, num - start);
        std::vector<float> xb(static_cast<std::size_t>(b) * input_dim);
        std::vector<float> tb(b);
        for (int i = 0; i < b; ++i) {
            const int src = start + i;
            for (int j = 0; j < input_dim; ++j) {
                xb[static_cast<std::size_t>(i) * input_dim + j] =
                    data.images[static_cast<std::size_t>(src) * input_dim + j];
            }
            tb[i] = static_cast<float>(data.labels[src]);
        }
        Tensor x(dev, {b, data.channels, data.rows, data.cols});
        x.upload(xb);
        Tensor logits = net.forward(x);
        std::vector<float> lg;
        logits.download(lg);
        for (int i = 0; i < b; ++i) {
            int best = 0;
            for (int c = 1; c < num_classes; ++c) {
                if (lg[static_cast<std::size_t>(i) * num_classes + c] >
                    lg[static_cast<std::size_t>(i) * num_classes + best]) {
                    best = c;
                }
            }
            if (best == static_cast<int>(tb[i])) ++correct;
            ++total;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(total);
}

}  // namespace

int main() {
    std::cout << std::unitbuf;  // 强制实时输出，避免管道缓冲导致长时间无输出
    auto dev = Device::create(Backend::OpenGL);
    if (!dev) {
        std::cerr << "OpenGL device creation failed\n";
        return 1;
    }
    std::cout << "[device] " << dev->info().name << "\n";

    const std::string data_dir = "D:/OpenDLL/data/CIFAR10/";
    const auto cifar_train =
        load_cifar10(data_dir + "train_images.bin", data_dir + "train_labels.bin", 3, 32, 32);
    const auto cifar_test =
        load_cifar10(data_dir + "test_images.bin", data_dir + "test_labels.bin", 3, 32, 32);

    Dataset train, val, test;
    split_622(cifar_train, cifar_test, 42, train, val, test);
    std::cout << "split 6:2:2 -> train " << train.labels.size() << " / val "
              << val.labels.size() << " / test " << test.labels.size() << "\n";

    ResNet net(*dev, 10);
    std::vector<Module*> modules;
    net.collect(modules);

    const int batch_size = 128;
    const float base_lr = 0.001f;
    const float momentum = 0.9f;
    SGDM sgd(base_lr / static_cast<float>(batch_size), momentum);

    const int num_train = static_cast<int>(train.labels.size());
    const int input_dim = train.channels * train.rows * train.cols;
    const int epochs = 10;

    float best_val_acc = -1.0f;
    int best_epoch = -1;

    const auto t_start = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // step decay：每 3 epoch 衰减 10 倍，压住 momentum 后期过冲
        float lr_mean = base_lr;
        for (int d = 0; d < epoch / 3; ++d) lr_mean *= 0.1f;
        sgd.set_lr(lr_mean / static_cast<float>(batch_size));

        net.train_mode(true);
        float total_loss = 0.0f;
        int num_batches = 0;

        for (int start = 0; start + batch_size <= num_train; start += batch_size) {
            std::vector<float> xb(static_cast<std::size_t>(batch_size) * input_dim);
            std::vector<float> tb(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                const int idx = start + i;
                for (int j = 0; j < input_dim; ++j) {
                    xb[static_cast<std::size_t>(i) * input_dim + j] =
                        train.images[static_cast<std::size_t>(idx) * input_dim + j];
                }
                tb[i] = static_cast<float>(train.labels[idx]);
            }

            Tensor x(*dev, {batch_size, train.channels, train.rows, train.cols});
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

            if (num_batches % 20 == 0) {
                std::cout << "  [epoch " << epoch << " batch " << num_batches << "/"
                          << (num_train / batch_size) << "] loss="
                          << (total_loss / static_cast<float>(num_batches)) << std::endl;
            }
        }

        const float avg_loss = total_loss / static_cast<float>(num_batches);
        std::cout << "epoch " << epoch << "  loss=" << avg_loss;

        // 每 2 epoch 在验证集上评估
        if ((epoch + 1) % 2 == 0 || epoch == epochs - 1) {
            net.train_mode(false);
            const float val_acc = evaluate(*dev, val, net, batch_size);
            std::cout << "  val_acc=" << val_acc;
            if (val_acc > best_val_acc) {
                best_val_acc = val_acc;
                best_epoch = epoch;
            }
        }
        std::cout << "\n";
    }
    const auto t_end = std::chrono::steady_clock::now();
    std::cout << "training time: "
              << std::chrono::duration<double>(t_end - t_start).count() << "s\n";

    // 取验证集上最好的 epoch，在测试集上评估
    std::cout << "\nbest epoch " << best_epoch << " (val_acc=" << best_val_acc << ")\n";
    net.train_mode(false);
    const float test_acc = evaluate(*dev, test, net, batch_size);
    std::cout << "test_acc=" << test_acc << "\n";

    return 0;
}
