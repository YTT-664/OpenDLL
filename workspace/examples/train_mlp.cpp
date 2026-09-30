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

float eval_accuracy(Device& dev, const Dataset& data, Linear& fc1, ReLU& relu, Linear& fc2,
                    int batch_size) {
    const int input_dim = data.rows * data.cols;
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

        Tensor x(dev, {batch_size, input_dim});
        x.upload(xb);
        Tensor logits = fc2.forward(relu.forward(fc1.forward(x)));

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
            if (best == static_cast<int>(tb[i])) {
                ++correct;
            }
        }
    }
    return static_cast<float>(correct) / static_cast<float>(num);
}

}  // namespace

int main() {
    auto dev = Device::create(Backend::OpenGL);
    if (!dev) {
        std::cerr << "OpenGL device creation failed\n";
        return 1;
    }

    const std::string data_dir = "D:/OpenDLL/data/FashionMNIST/raw/";
    const auto train = load_mnist(data_dir + "train-images-idx3-ubyte",
                                  data_dir + "train-labels-idx1-ubyte");
    const auto test = load_mnist(data_dir + "t10k-images-idx3-ubyte",
                                 data_dir + "t10k-labels-idx1-ubyte");

    std::cout << "train samples: " << train.labels.size() << "\n";
    std::cout << "test samples: " << test.labels.size() << "\n";
    std::cout << "input dim: " << (train.rows * train.cols)
              << "  classes: " << train.num_classes << "\n";

    Linear fc1(*dev, 784, 128);
    ReLU relu(*dev);
    Linear fc2(*dev, 128, 10);

    std::vector<Module*> modules = {&fc1, &relu, &fc2};

    const int batch_size = 64;
    const float lr = 0.1f;
    SGD sgd(lr / static_cast<float>(batch_size));

    const int num_train = static_cast<int>(train.labels.size());
    const int input_dim = train.rows * train.cols;

    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < 20; ++epoch) {
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

            Tensor x(*dev, {batch_size, input_dim});
            x.upload(xb);
            Tensor t(*dev, {batch_size});
            t.upload(tb);

            // forward
            Tensor h = fc1.forward(x);
            Tensor a = relu.forward(h);
            Tensor logits = fc2.forward(a);

            // loss（sum，未平均）
            Tensor loss = cross_entropy(*dev, logits, t);
            std::vector<float> lv;
            loss.download(lv);
            float batch_loss = 0.0f;
            for (float v : lv) {
                batch_loss += v;
            }
            total_loss += batch_loss / static_cast<float>(batch_size);
            ++num_batches;

            // backward
            Tensor grad_logits = cross_entropy_backward(*dev, logits, t);
            Tensor grad_a = fc2.backward(grad_logits);
            Tensor grad_h = relu.backward(grad_a);
            fc1.backward(grad_h);

            // update
            sgd.step(modules);
        }

        const float acc = eval_accuracy(*dev, test, fc1, relu, fc2, batch_size);
        std::cout << "epoch " << epoch << "  loss=" << (total_loss / num_batches)
                  << "  test_acc=" << acc << "\n";
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "training time: " << seconds << "s\n";

    return 0;
}
