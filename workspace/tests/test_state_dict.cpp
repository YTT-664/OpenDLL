#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "opendll/device.hpp"
#include "opendll/models/resnet.hpp"
#include "opendll/state_dict.hpp"

using namespace opendll;

int main() {
    auto dev = Device::create(Backend::OpenGL);
    if (!dev) {
        std::cerr << "OpenGL device creation failed\n";
        return 1;
    }
    bool all_ok = true;

    // 1. ResNet18 分类模型（10 类）：验证手动命名 + 条目完整性
    {
        ResNet18 net(*dev, 32, 10);
        auto state = net.collect_state();

        int num_params = 0, num_buffers = 0;
        for (const auto& e : state) {
            if (e.is_buffer) {
                ++num_buffers;
            } else {
                ++num_params;
            }
        }
        std::cout << "state_dict entries: " << state.size() << " (params=" << num_params
                  << ", buffers=" << num_buffers << ")\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(state.size(), 6); ++i) {
            std::cout << "  " << state[i].name << (state[i].is_buffer ? " [buffer]" : "")
                      << "\n";
        }

        // ResNet18：20 个 conv(weight+bias) + 20 个 bn(gamma+beta) + 1 fc(weight+bias)
        // 可训练参数 = 20*2 + 20*2 + 2 = 82；buffers = 20*2 = 40（bn running stats）
        const bool ok = (num_params == 82 && num_buffers == 40);
        std::cout << "entry counts (82 params / 40 buffers): " << (ok ? "PASS" : "FAIL")
                  << "\n";
        all_ok = all_ok && ok;

        // 2. save + load roundtrip
        save_state_dict("test_state.bin", state);

        ResNet18 net2(*dev, 32, 10);
        auto state2 = net2.collect_state();
        load_state_dict("test_state.bin", state2);

        bool match = true;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (state[i].name != state2[i].name) {
                match = false;
                break;
            }
            std::vector<float> a, b;
            state[i].tensor->download(a);
            state2[i].tensor->download(b);
            if (a != b) {
                match = false;
                break;
            }
        }
        std::cout << "save/load roundtrip (all tensors identical): "
                  << (match ? "PASS" : "FAIL") << "\n";
        all_ok = all_ok && match;
    }

    // 3. num_classes=0：无分类头，输出 [N, 512] 特征向量
    {
        ResNet18 feat(*dev, 32, 0);
        auto state = feat.collect_state();
        // 无 fc，所以可训练参数 = 80（少了 fc.weight/bias），buffers 不变
        int num_params = 0;
        for (const auto& e : state) {
            if (!e.is_buffer) ++num_params;
        }
        const bool ok = (num_params == 80);
        std::cout << "backbone only params (80): " << (ok ? "PASS" : "FAIL") << "\n";
        all_ok = all_ok && ok;

        Tensor x(*dev, {2, 3, 32, 32});
        std::vector<float> xd(2 * 3 * 32 * 32, 0.1f);
        x.upload(xd);
        Tensor out = feat.forward(x);
        const bool shape_ok = (out.ndim() == 2 && out.dim(0) == 2 && out.dim(1) == 512);
        std::cout << "backbone forward -> [" << out.dim(0) << ", " << out.dim(1) << "]: "
                  << (shape_ok ? "PASS" : "FAIL") << "\n";
        all_ok = all_ok && shape_ok;
    }

    std::cout << (all_ok ? "ALL STATE_DICT TESTS PASSED" : "SOME TESTS FAILED") << "\n";
    return all_ok ? 0 : 1;
}
