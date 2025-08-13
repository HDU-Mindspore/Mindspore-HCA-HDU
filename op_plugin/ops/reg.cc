/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include "ops/generated_reg.h"

namespace op_plugin {

extern "C" bool IsKernelRegistered(const char *op_name) {
  return std::find(register_op_name.begin(), register_op_name.end(), op_name) != register_op_name.end();
}

extern "C" int GetRegisteredOpCount() {
  return static_cast<int>(register_op_name.size());
}

extern "C" const char** GetAllRegisteredOps() {
  static std::vector<const char*> op_names;
  static bool initialized = false;
  if (!initialized) {
    op_names.reserve(register_op_name.size());
    std::transform(register_op_name.begin(), register_op_name.end(), std::back_inserter(op_names),
                   [](const std::string& name) { return name.c_str(); });
    initialized = true;
  }
  return op_names.data();
}
}  // namespace op_plugin
