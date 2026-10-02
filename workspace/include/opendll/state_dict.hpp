#pragma once

#include <string>
#include <vector>

#include "opendll/module.hpp"

namespace opendll {

// 把 state_dict 条目序列化到二进制文件。
// 格式：magic "ODLL" + version(u32) + count(u32)，随后每个条目：
//   name_len(u32) + name + numel(u64) + float32 数据。
void save_state_dict(const std::string& path, const std::vector<StateEntry>& entries);

// 从文件读取并按 name 匹配填充 entries 里的张量。
// 缺项（文件里没有某个 name）或 numel 不匹配时抛 std::runtime_error。
void load_state_dict(const std::string& path, const std::vector<StateEntry>& entries);

}  // namespace opendll
