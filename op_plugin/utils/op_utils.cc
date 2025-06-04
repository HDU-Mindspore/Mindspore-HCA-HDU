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
#include <torch/extension.h>  // 头文件引用部分

int8_t GetDtype(const std::string &dtypes) {
  int8_t type = 6;
  std::unordered_map<std::string, int8_t> m{{"uint8", 0}, {"int8", 1},    {"int16", 2},   {"int32", 3},
                                            {"int64", 4}, {"float16", 5}, {"float32", 6}, {"float64", 7}};
  if (m.count(dtypes)) {
    type = m[dtypes];
  }
  return type;
}

std::vector<at::Tensor> ConvertToATenTensors(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                                c10::DeviceType device_type) {
  std::vector<at::Tensor> tensors;
  for (int i = 0; i < nparam; i++) {
    std::vector<int64_t> size;
    for (int j = 0; j < ndims[i]; j++) {
      size.push_back(shapes[i][j]);
    }
    int8_t type = GetDtype(dtypes[i]);
    // 注意：这里device设置为kCPU时跑的CPU算子，设置为kCUDA时跑的GPU算子，其他device不在使用范围内
    auto option = at::TensorOptions().dtype(static_cast<c10::ScalarType>(type)).device(device_type);
    tensors.emplace_back(at::from_blob(params[i], size, option));
  }
  return tensors;
}

/*
at::Scalar ConvertToATenScalar(const ScalarPtr &scalar) {
  MS_EXCEPTION_IF_NULL(scalar);

  TypePtr data_type = scalar->type();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeBool:
      return at::Scalar(GetValue<bool>(scalar));
    case kNumberTypeInt8:
      return at::Scalar(GetValue<int8_t>(scalar));
    case kNumberTypeInt16:
      return at::Scalar(GetValue<int16_t>(scalar));
    case kNumberTypeInt32:
      return at::Scalar(GetValue<int32_t>(scalar));
    case kNumberTypeInt64:
      return at::Scalar(GetValue<int64_t>(scalar));
    case kNumberTypeUInt8:
      return at::Scalar(GetValue<uint8_t>(scalar));
    case kNumberTypeUInt16:
      return at::Scalar(GetValue<uint16_t>(scalar));
    case kNumberTypeUInt32:
      return at::Scalar(GetValue<uint32_t>(scalar));
    case kNumberTypeUInt64:
      return at::Scalar(GetValue<uint64_t>(scalar));
    case kNumberTypeFloat32:
      return at::Scalar(GetValue<float>(scalar));
    case kNumberTypeFloat64:
      return at::Scalar(GetValue<double>(scalar));
    default:
      MS_LOG(EXCEPTION) << "When convert scalar to tensor, the scalar type: " << data_type << " is invalid.";
  }
}
*/

void output_memcpy(void *output, const torch::Tensor &t) { memcpy(output, t.data_ptr(), t.element_size() * t.numel()); }
