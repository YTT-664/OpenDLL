#include "opendll/models/resnet.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opendll/ops.hpp"

namespace opendll {

namespace {

// ResNet18 的 BasicBlock：两个 3x3 卷积 + 残差连接（stride>1 或通道变化时带 downsample）。
class BasicBlock final : public Module {
public:
    BasicBlock(Device& dev, std::string prefix, int in_planes, int planes, int stride)
        : dev_(dev),
          conv1_(dev, in_planes, planes, 3, stride, 1),
          bn1_(dev, planes),
          relu1_(dev),
          conv2_(dev, planes, planes, 3, 1, 1),
          bn2_(dev, planes),
          relu2_(dev),
          downsample_(stride != 1 || in_planes != planes),
          ds_conv_(dev, in_planes, planes, 1, stride, 0),
          ds_bn_(dev, planes) {
        // 手动命名（PyTorch 风格）
        set_name(prefix);
        conv1_.set_name(prefix + ".conv1");
        bn1_.set_name(prefix + ".bn1");
        conv2_.set_name(prefix + ".conv2");
        bn2_.set_name(prefix + ".bn2");
        if (downsample_) {
            ds_conv_.set_name(prefix + ".downsample.0");
            ds_bn_.set_name(prefix + ".downsample.1");
        }
    }

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

    std::vector<StateEntry> collect_state() override {
        std::vector<StateEntry> out;
        auto merge = [&out](Module& m) {
            auto s = m.collect_state();
            out.insert(out.end(), s.begin(), s.end());
        };
        merge(conv1_);
        merge(bn1_);
        merge(conv2_);
        merge(bn2_);
        if (downsample_) {
            merge(ds_conv_);
            merge(ds_bn_);
        }
        return out;
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

}  // namespace

struct ResNet18::Impl {
    Device& dev;
    Conv2d conv1;
    BatchNorm2d bn1;
    ReLU relu;
    std::vector<std::unique_ptr<BasicBlock>> layer1, layer2, layer3, layer4;
    std::unique_ptr<Linear> fc;  // num_classes==0 时为 null
    int64_t last_H_ = 0, last_W_ = 0;

    Impl(Device& d, int num_classes)
        : dev(d), conv1(d, 3, 64, 3, 1, 1), bn1(d, 64), relu(d) {
        conv1.set_name("conv1");
        bn1.set_name("bn1");
        relu.set_name("relu");
        make_layer(layer1, "layer1", 64, 64, 2, 1);
        make_layer(layer2, "layer2", 64, 128, 2, 2);
        make_layer(layer3, "layer3", 128, 256, 2, 2);
        make_layer(layer4, "layer4", 256, 512, 2, 2);
        if (num_classes > 0) {
            fc = std::make_unique<Linear>(d, 512, num_classes);
            fc->set_name("fc");
        }
    }

    void make_layer(std::vector<std::unique_ptr<BasicBlock>>& layer, const std::string& name,
                    int in_planes, int planes, int blocks, int stride) {
        layer.push_back(
            std::make_unique<BasicBlock>(dev, name + ".0", in_planes, planes, stride));
        for (int i = 1; i < blocks; ++i) {
            layer.push_back(
                std::make_unique<BasicBlock>(dev, name + "." + std::to_string(i), planes,
                                             planes, 1));
        }
    }

    Tensor forward(const Tensor& x) {
        Tensor out = relu.forward(bn1.forward(conv1.forward(x)));
        for (auto& b : layer1) out = b->forward(out);
        for (auto& b : layer2) out = b->forward(out);
        for (auto& b : layer3) out = b->forward(out);
        for (auto& b : layer4) out = b->forward(out);
        last_H_ = out.dim(2);
        last_W_ = out.dim(3);
        out = avgpool2d(dev, out, static_cast<int>(last_H_), static_cast<int>(last_H_), 0);
        out = out.view({out.dim(0), out.dim(1)});
        if (fc) {
            return fc->forward(out);
        }
        return out;  // [N, 512] 特征向量
    }

    Tensor backward(const Tensor& grad_out) {
        Tensor grad = fc ? fc->backward(grad_out) : grad_out;  // [N, 512]
        grad = grad.view({grad.dim(0), grad.dim(1), 1, 1});
        grad = avgpool2d_backward(dev, grad, static_cast<int>(last_H_),
                                  static_cast<int>(last_W_));
        for (auto it = layer4.rbegin(); it != layer4.rend(); ++it) grad = (*it)->backward(grad);
        for (auto it = layer3.rbegin(); it != layer3.rend(); ++it) grad = (*it)->backward(grad);
        for (auto it = layer2.rbegin(); it != layer2.rend(); ++it) grad = (*it)->backward(grad);
        for (auto it = layer1.rbegin(); it != layer1.rend(); ++it) grad = (*it)->backward(grad);
        grad = relu.backward(grad);
        grad = bn1.backward(grad);
        return conv1.backward(grad);
    }

    std::vector<std::pair<Tensor*, Tensor*>> parameters() {
        std::vector<std::pair<Tensor*, Tensor*>> p;
        auto add_all = [&p](Module& m) {
            auto mp = m.parameters();
            p.insert(p.end(), mp.begin(), mp.end());
        };
        add_all(conv1);
        add_all(bn1);
        for (auto& b : layer1) add_all(*b);
        for (auto& b : layer2) add_all(*b);
        for (auto& b : layer3) add_all(*b);
        for (auto& b : layer4) add_all(*b);
        if (fc) add_all(*fc);
        return p;
    }

    std::vector<StateEntry> collect_state() {
        std::vector<StateEntry> out;
        auto merge = [&out](Module& m) {
            auto s = m.collect_state();
            out.insert(out.end(), s.begin(), s.end());
        };
        merge(conv1);
        merge(bn1);
        for (auto& b : layer1) merge(*b);
        for (auto& b : layer2) merge(*b);
        for (auto& b : layer3) merge(*b);
        for (auto& b : layer4) merge(*b);
        if (fc) merge(*fc);
        return out;
    }

    void train_mode(bool on) {
        bn1.train(on);
        for (auto& b : layer1) b->train_mode(on);
        for (auto& b : layer2) b->train_mode(on);
        for (auto& b : layer3) b->train_mode(on);
        for (auto& b : layer4) b->train_mode(on);
    }
};

ResNet18::ResNet18(Device& dev, int in_size, int num_classes)
    : impl_(std::make_unique<Impl>(dev, num_classes)),
      in_size_(in_size),
      num_classes_(num_classes) {}

ResNet18::~ResNet18() = default;

Tensor ResNet18::forward(const Tensor& x) {
    return impl_->forward(x);
}

Tensor ResNet18::backward(const Tensor& grad_out) {
    return impl_->backward(grad_out);
}

std::vector<std::pair<Tensor*, Tensor*>> ResNet18::parameters() {
    return impl_->parameters();
}

std::vector<StateEntry> ResNet18::collect_state() {
    return impl_->collect_state();
}

void ResNet18::train_mode(bool on) {
    impl_->train_mode(on);
}

}  // namespace opendll
