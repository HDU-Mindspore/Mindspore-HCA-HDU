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
#include <string.h>
#include <torch/extension.h>
#include <iostream>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int Index(int nparam, void **params, int *ndims, int64_t **shapes,
                     const char **dtypes, void *stream, void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto self = tensors[0];
  auto at_output = tensors[nparam - 1];

  c10::List<c10::optional<at::Tensor>> at_indices;
  for (auto it = tensors.begin() + 1; it != tensors.end() - 1; ++it) {
    at_indices.push_back(*it);
  }

  at::index_out(at_output, self, at_indices);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
