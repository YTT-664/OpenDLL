#include "opendll/state_dict.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "opendll/tensor.hpp"

namespace opendll {

namespace {

void write_u32(std::ofstream& f, std::uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_u64(std::ofstream& f, std::uint64_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

std::uint32_t read_u32(std::ifstream& f) {
    std::uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

std::uint64_t read_u64(std::ifstream& f) {
    std::uint64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

}  // namespace

void save_state_dict(const std::string& path, const std::vector<StateEntry>& entries) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open for write: " + path);
    }

    f.write("ODLL", 4);
    write_u32(f, 1u);  // version
    write_u32(f, static_cast<std::uint32_t>(entries.size()));

    for (const auto& e : entries) {
        const std::size_t numel = e.tensor->numel();
        std::vector<float> data(numel);
        e.tensor->download(data);

        write_u32(f, static_cast<std::uint32_t>(e.name.size()));
        f.write(e.name.data(), static_cast<std::streamsize>(e.name.size()));
        write_u64(f, static_cast<std::uint64_t>(numel));
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
    }
}

void load_state_dict(const std::string& path, const std::vector<StateEntry>& entries) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open for read: " + path);
    }

    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (std::memcmp(magic, "ODLL", 4) != 0) {
        throw std::runtime_error("bad state_dict magic: " + path);
    }
    const std::uint32_t version = read_u32(f);
    (void)version;
    const std::uint32_t count = read_u32(f);

    std::unordered_map<std::string, std::vector<float>> data;
    data.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t name_len = read_u32(f);
        std::string name(name_len, '\0');
        f.read(name.data(), static_cast<std::streamsize>(name_len));
        const std::uint64_t numel = read_u64(f);
        std::vector<float> d(static_cast<std::size_t>(numel));
        f.read(reinterpret_cast<char*>(d.data()),
               static_cast<std::streamsize>(numel * sizeof(float)));
        data.emplace(std::move(name), std::move(d));
    }

    for (const auto& e : entries) {
        auto it = data.find(e.name);
        if (it == data.end()) {
            throw std::runtime_error("state_dict missing entry: " + e.name);
        }
        if (it->second.size() != e.tensor->numel()) {
            throw std::runtime_error("state_dict numel mismatch for: " + e.name);
        }
        e.tensor->upload(it->second);
    }
}

}  // namespace opendll
