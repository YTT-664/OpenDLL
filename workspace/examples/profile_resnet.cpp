#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opendll/dataset.hpp"
#include "opendll/device.hpp"
#include "opendll/module.hpp"
#include "opendll/ops.hpp"
#include "opendll/optimizer.hpp"

using namespace opendll;
using Clock = std::chrono::steady_clock;

namespace {

// ---- 埋点统计 ----
struct Stat {
    double ms = 0.0;      // 累计耗时
    double mflops = 0.0;  // 理论计算量（百万 FLOPs）
};

std::vector<std::string> order;  // 首次出现顺序（保证输出有序）
std::map<std::string, Stat> g_stats;
Clock::time_point g_t0;

void tic() { g_t0 = Clock::now(); }

void toc(const std::string& name, double mflops) {
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - g_t0).count();
    auto it = g_stats.find(name);
    if (it == g_stats.end()) {
        g_stats[name] = Stat{ms, mflops};
        order.push_back(name);
    } else {
        it->second.ms += ms;
        it->second.mflops = mflops;
    }
}

// ---- 理论 FLOPs 计算（单位：百万 FLOPs = MFLOPs）----
double conv_mflops(int64_t N, int64_t Cout, int64_t Hout, int64_t Wout, int64_t Cin,
                   int64_t KH, int64_t KW) {
    // 每次乘加算 2 FLOPs
    return 2.0 * static_cast<double>(N) * Cout * Hout * Wout * Cin * KH * KW / 1e6;
}
double linear_mflops(int64_t M, int64_t N, int64_t K) {
    return 2.0 * static_cast<double>(M) * N * K / 1e6;
}
double bn_mflops(int64_t N, int64_t C, int64_t H, int64_t W) {
    // mean + var + normalize + scale/offset，近似 4 FLOPs/element
    return 4.0 * static_cast<double>(N) * C * H * W / 1e6;
}
double ew_mflops(int64_t numel) {  // relu / add 等逐元素
    return static_cast<double>(numel) / 1e6;
}

// ---- ResNet18（CIFAR 变体，带埋点）----
class BasicBlock final : public Module {
public:
    BasicBlock(Device& dev, std::string prefix, int in_planes, int planes, int stride)
        : dev_(dev),
          prefix_(std::move(prefix)),
          in_planes_(in_planes),
          planes_(planes),
          stride_(stride),
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
        const int64_t N = x.dim(0), Cin = x.dim(1), H = x.dim(2), W = x.dim(3);
        in_H_ = H;
        in_W_ = W;
        const int64_t H1 = (H + 2 - 3) / stride_ + 1;  // padding=1, kernel=3
        const int64_t W1 = (W + 2 - 3) / stride_ + 1;
        out_H_ = H1;
        out_W_ = W1;

        double mf;
        Tensor out;

        mf = conv_mflops(N, planes_, H1, W1, Cin, 3, 3);
        tic(); out = conv1_.forward(x); dev_.synchronize(); toc(prefix_ + ".conv1", mf);

        mf = bn_mflops(N, planes_, H1, W1);
        tic(); out = bn1_.forward(out); dev_.synchronize(); toc(prefix_ + ".bn1", mf);

        mf = ew_mflops(N * planes_ * H1 * W1);
        tic(); out = relu1_.forward(out); dev_.synchronize(); toc(prefix_ + ".relu1", mf);

        mf = conv_mflops(N, planes_, H1, W1, planes_, 3, 3);
        tic(); out = conv2_.forward(out); dev_.synchronize(); toc(prefix_ + ".conv2", mf);

        mf = bn_mflops(N, planes_, H1, W1);
        tic(); out = bn2_.forward(out); dev_.synchronize(); toc(prefix_ + ".bn2", mf);

        Tensor shortcut = x;
        if (downsample_) {
            mf = conv_mflops(N, planes_, H1, W1, Cin, 1, 1);
            tic(); shortcut = ds_conv_.forward(x); dev_.synchronize(); toc(prefix_ + ".ds_conv", mf);

            mf = bn_mflops(N, planes_, H1, W1);
            tic(); shortcut = ds_bn_.forward(shortcut); dev_.synchronize(); toc(prefix_ + ".ds_bn", mf);
        }

        mf = ew_mflops(N * planes_ * H1 * W1);
        tic(); out = add(dev_, out, shortcut); dev_.synchronize(); toc(prefix_ + ".add", mf);

        mf = ew_mflops(N * planes_ * H1 * W1);
        tic(); out = relu2_.forward(out); dev_.synchronize(); toc(prefix_ + ".relu2", mf);

        return out;
    }

    Tensor backward(const Tensor& grad_out) override {
        const int64_t N = grad_out.dim(0), C = grad_out.dim(1), H1 = out_H_, W1 = out_W_;

        double mf;
        Tensor grad;

        mf = ew_mflops(N * C * H1 * W1);
        tic(); grad = relu2_.backward(grad_out); dev_.synchronize(); toc(prefix_ + ".relu2.bwd", mf);

        mf = bn_mflops(N, C, H1, W1) * 2;
        tic(); Tensor g = bn2_.backward(grad); dev_.synchronize(); toc(prefix_ + ".bn2.bwd", mf);

        mf = conv_mflops(N, C, H1, W1, C, 3, 3) * 2;  // grad_input + grad_weight
        tic(); g = conv2_.backward(g); dev_.synchronize(); toc(prefix_ + ".conv2.bwd", mf);

        mf = ew_mflops(N * C * H1 * W1);
        tic(); g = relu1_.backward(g); dev_.synchronize(); toc(prefix_ + ".relu1.bwd", mf);

        mf = bn_mflops(N, C, H1, W1) * 2;
        tic(); g = bn1_.backward(g); dev_.synchronize(); toc(prefix_ + ".bn1.bwd", mf);

        mf = conv_mflops(N, in_planes_, in_H_, in_W_, C, 3, 3) * 2;
        tic(); Tensor grad_x = conv1_.backward(g); dev_.synchronize(); toc(prefix_ + ".conv1.bwd", mf);

        if (downsample_) {
            mf = bn_mflops(N, C, H1, W1) * 2;
            tic(); Tensor gs = ds_bn_.backward(grad); dev_.synchronize(); toc(prefix_ + ".ds_bn.bwd", mf);

            mf = conv_mflops(N, in_planes_, in_H_, in_W_, C, 1, 1) * 2;
            tic(); gs = ds_conv_.backward(gs); dev_.synchronize(); toc(prefix_ + ".ds_conv.bwd", mf);

            mf = ew_mflops(N * in_planes_ * in_H_ * in_W_);
            tic(); grad_x = add(dev_, grad_x, gs); dev_.synchronize(); toc(prefix_ + ".add.bwd", mf);
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
    std::string prefix_;
    int in_planes_;
    int planes_;
    int stride_;
    int64_t in_H_ = 0, in_W_ = 0, out_H_ = 0, out_W_ = 0;
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
    int num_classes;

    ResNet(Device& d, int nc = 10)
        : dev(d), conv1(d, 3, 64, 3, 1, 1), bn1(d, 64), relu(d), fc(d, 512, nc),
          num_classes(nc) {
        make_layer(layer1, "layer1", 64, 64, 2, 1);
        make_layer(layer2, "layer2", 64, 128, 2, 2);
        make_layer(layer3, "layer3", 128, 256, 2, 2);
        make_layer(layer4, "layer4", 256, 512, 2, 2);
    }

    void make_layer(std::vector<BasicBlock>& layer, const std::string& name, int in_planes,
                    int planes, int blocks, int stride) {
        layer.emplace_back(dev, name + ".0", in_planes, planes, stride);
        for (int i = 1; i < blocks; ++i) {
            layer.emplace_back(dev, name + "." + std::to_string(i), planes, planes, 1);
        }
    }

    Tensor forward(const Tensor& x) {
        const int64_t N = x.dim(0), Cin = x.dim(1), H = x.dim(2), W = x.dim(3);
        double mf;
        Tensor out;

        mf = conv_mflops(N, 64, H, W, Cin, 3, 3);
        tic(); out = conv1.forward(x); dev.synchronize(); toc("conv1", mf);

        mf = bn_mflops(N, 64, H, W);
        tic(); out = bn1.forward(out); dev.synchronize(); toc("bn1", mf);

        mf = ew_mflops(N * 64 * H * W);
        tic(); out = relu.forward(out); dev.synchronize(); toc("relu", mf);

        for (auto& b : layer1) out = b.forward(out);
        for (auto& b : layer2) out = b.forward(out);
        for (auto& b : layer3) out = b.forward(out);
        for (auto& b : layer4) out = b.forward(out);

        const int64_t Hh = out.dim(2);
        mf = ew_mflops(N * out.dim(1) * Hh * Hh);
        tic(); out = avgpool2d(dev, out, static_cast<int>(Hh), static_cast<int>(Hh), 0);
        dev.synchronize(); toc("avgpool", mf);

        out = out.view({out.dim(0), out.dim(1)});
        mf = linear_mflops(N, num_classes, 512);
        tic(); out = fc.forward(out); dev.synchronize(); toc("fc", mf);
        return out;
    }

    Tensor backward(const Tensor& grad_out) {
        const int64_t N = grad_out.dim(0);
        double mf;
        Tensor grad;

        mf = linear_mflops(N, num_classes, 512) * 2;
        tic(); grad = fc.backward(grad_out); dev.synchronize(); toc("fc.bwd", mf);

        grad = grad.view({grad.dim(0), grad.dim(1), 1, 1});
        for (auto it = layer4.rbegin(); it != layer4.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer3.rbegin(); it != layer3.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer2.rbegin(); it != layer2.rend(); ++it) grad = it->backward(grad);
        for (auto it = layer1.rbegin(); it != layer1.rend(); ++it) grad = it->backward(grad);

        const int64_t C = grad.dim(1), H = grad.dim(2), W = grad.dim(3);
        mf = ew_mflops(N * C * H * W);
        tic(); grad = relu.backward(grad); dev.synchronize(); toc("relu.bwd", mf);

        mf = bn_mflops(N, C, H, W) * 2;
        tic(); grad = bn1.backward(grad); dev.synchronize(); toc("bn1.bwd", mf);

        mf = conv_mflops(N, 3, H, W, C, 3, 3) * 2;
        tic(); grad = conv1.backward(grad); dev.synchronize(); toc("conv1.bwd", mf);
        return grad;
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

}  // namespace

int main() {
    auto dev = Device::create(Backend::OpenGL);
    if (!dev) {
        std::cerr << "OpenGL device creation failed\n";
        return 1;
    }
    std::cout << "[device] " << dev->info().name << "\n";

    const std::string data_dir = "D:/OpenDLL/data/CIFAR10/";
    const auto train = load_cifar10(data_dir + "train_images.bin", data_dir + "train_labels.bin",
                                    3, 32, 32);

    const int batch_size = 128;
    const int input_dim = 3 * 32 * 32;

    ResNet net(*dev, 10);
    std::vector<Module*> modules;
    net.collect(modules);

    // 构造一个 batch
    std::vector<float> xb(static_cast<std::size_t>(batch_size) * input_dim);
    std::vector<float> tb(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            xb[static_cast<std::size_t>(i) * input_dim + j] =
                train.images[static_cast<std::size_t>(i) * input_dim + j];
        }
        tb[i] = static_cast<float>(train.labels[i]);
    }
    Tensor x(*dev, {batch_size, 3, 32, 32});
    x.upload(xb);
    Tensor t(*dev, {batch_size});
    t.upload(tb);

    // forward + loss + backward
    Tensor logits = net.forward(x);
    Tensor loss = cross_entropy(*dev, logits, t);
    Tensor grad_logits = cross_entropy_backward(*dev, logits, t);
    net.backward(grad_logits);

    // 参数更新
    double total_params = 0.0;
    for (auto* m : modules) {
        for (auto& [p, g] : m->parameters()) {
            total_params += static_cast<double>(p->numel());
        }
    }
    SGD sgd(0.01f / static_cast<float>(batch_size));
    tic();
    sgd.step(modules);
    dev->synchronize();
    toc("sgd_update", 2.0 * total_params / 1e6);  // 每参数乘 + 减 = 2 FLOPs

    // 输出统计
    std::cout << "\n"
              << std::setw(22) << std::left << "op" << std::setw(12) << std::right << "MFLOPs"
              << std::setw(10) << "ms" << std::setw(14) << "GFLOPs/s" << "\n";
    std::cout << std::string(58, '-') << "\n";

    double total_ms = 0.0, total_mflops = 0.0;
    for (const auto& name : order) {
        const auto& s = g_stats[name];
        total_ms += s.ms;
        total_mflops += s.mflops;
        const double gflops_s = s.ms > 0.0 ? (s.mflops / 1000.0) / (s.ms / 1000.0) : 0.0;
        std::cout << std::setw(22) << std::left << name << std::setw(12) << std::right
                  << std::fixed << std::setprecision(2) << s.mflops << std::setw(10)
                  << std::setprecision(3) << s.ms << std::setw(14) << std::setprecision(1)
                  << gflops_s << "\n";
    }
    std::cout << std::string(58, '-') << "\n";
    std::cout << std::setw(22) << std::left << "TOTAL" << std::setw(12) << std::right
              << std::fixed << std::setprecision(2) << total_mflops << std::setw(10)
              << std::setprecision(3) << total_ms << std::setw(14) << std::setprecision(1)
              << (total_mflops / 1000.0) / (total_ms / 1000.0) << "\n";

    return 0;
}
