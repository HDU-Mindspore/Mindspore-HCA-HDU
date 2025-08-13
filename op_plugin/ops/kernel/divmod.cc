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
#include "ops/op_enum.h"

namespace op_plugin {
namespace aten_op {
extern "C" int DivMod(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                   void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input1 = tensors[0];
  auto at_input2 = tensors[1];
  auto at_output = tensors[nparam - 1];

  static const c10::optional<c10::string_view> floor("floor");
  static const c10::optional<c10::string_view> trunc("trunc");
  KernelInputInfo& input_info = *static_cast<KernelInputInfo*>(extra);
  KernelInputUtils input_utils(input_info);
  c10::optional<c10::string_view> pt_rounding_mode = c10::nullopt;
  if (!input_utils.IsNoneInput(2)) {
    auto rounding_mode = input_utils.GetKernelInput<int64_t>(2);
    switch (rounding_mode) {
      case RoundingMode::TRUNC:
        pt_rounding_mode = trunc;
        break;
      case RoundingMode::FLOOR:
        pt_rounding_mode = floor;
        break;
      default:
        throw std::runtime_error("Unsupported rounding_mode num:" + std::to_string(rounding_mode) + ".");
        break;
    }
  }
  at::div_out(at_output, at_input1, at_input2, pt_rounding_mode);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
