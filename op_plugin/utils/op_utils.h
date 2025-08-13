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


#ifndef MS_PLUGIN_OP_UTILS_H_
#define MS_PLUGIN_OP_UTILS_H_
#include <torch/extension.h>

#include <string>
#include <vector>

namespace op_plugin {
/// \brief TypeId defines data type identifiers.
enum TypeId : int {
  kTypeUnknown = 0,
  //
  // Meta types.
  //
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,
  kMetaTypeAny,
  kMetaTypeObject,
  kMetaTypeTypeType,
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeRowTensorType,
  kObjectTypeCOOTensorType,
  kObjectTypeUndeterminedType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeBFloat16,
  kNumberTypeDouble,
  kNumberTypeComplex,
  kNumberTypeComplex64,
  kNumberTypeComplex128,
  kNumberTypeInt4,
  kNumberTypeGLUInt,
  kNumberTypeEnd,
  //
  // Monad Types
  //
  kMonadTypeBegin = kNumberTypeEnd,
  kObjectTypeMonad,
  kObjectTypeUMonad,
  kObjectTypeIOMonad,
  kMonadTypeEnd,
  //
  // Sparse Types
  //
  kSparseTypeBegin = kMonadTypeEnd,
  kObjectTypeCSRTensorType,
  kObjectTypeSparseTensorType,
  kObjectTypeMapTensorType,
  kSparseTypeEnd,
  // New types should placed at the end of enum,
  // in order to keep fit with the type of existing model on the lite side.
};

// 将mindspore的数据类型转化为pytorch的标准数据类型序号
int8_t GetDtype(const std::string &dtypes);

// 将 mindspore kernel 的 inputs/outputs 转换为 pytorch 的 tensor
std::vector<at::Tensor> ConvertToATenTensors(int nparam, void **params, int *ndims, int64_t **shapes,
                                                const char **dtypes, c10::DeviceType device_type = c10::kCPU);

// 将入参没有输出的pytorch 算子的计算结果拷贝到kernel的输出内存
void output_memcpy(void *output, const torch::Tensor &t);

class CustomKernelData {
 public:
  CustomKernelData() = default;
  virtual ~CustomKernelData() = default;
};

// KernelInputInfo is an interface class.
// There is also a copy of the same code in the MindSpore repository.
// Both sides should be consistent and neither side's code should be modified separately.
class KernelInputInfo {
 public:
  KernelInputInfo() = default;
  virtual ~KernelInputInfo() = default;
  virtual bool IsScalarInput(size_t idx) = 0;

  template <typename T>
  inline T GetKernelInput(size_t) const {
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  void SetKernelData(CustomKernelData *kernel_data) { kernel_data_ = kernel_data; }
  const CustomKernelData *KernelData() const { return kernel_data_; }

  void DestructKernelData() {
    delete kernel_data_;
    kernel_data_ = nullptr;
  }
  virtual size_t GetInputSize() = 0;

  virtual bool GetBoolInput(size_t idx) = 0;
  virtual int64_t GetIntInput(size_t idx) = 0;
  virtual float GetFloatInput(size_t idx) = 0;
  virtual std::string GetStrInput(size_t idx) = 0;

  virtual std::vector<int64_t> GetIntVecInput(size_t idx) = 0;
  virtual std::vector<float> GetFloatVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<float>> GetFloat2DVecInput(size_t idx) = 0;
  virtual int GetInputTypeId(size_t idx) = 0;
  std::vector<size_t> workspace_;

 private:
  CustomKernelData *kernel_data_{nullptr};
};

class KernelInputUtils {
 public:
  explicit KernelInputUtils(KernelInputInfo &input_info) : input_info_(input_info) {}
  ~KernelInputUtils() = default;
  bool IsNoneInput(size_t idx);
  bool IsScalarInput(size_t idx);

  template <typename T>
  inline T GetKernelInput(size_t idx) {
    return T();
  }

 private:
  KernelInputInfo& input_info_;
};

template <>
inline bool KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetBoolInput(idx);
}

template <>
inline c10::optional<bool> KernelInputUtils::GetKernelInput(size_t idx) {
  if (IsNoneInput(idx)) {
    return c10::nullopt;
  }
  return input_info_.GetBoolInput(idx);
}

template <>
inline int64_t KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetIntInput(idx);
}

template <>
inline c10::optional<int64_t> KernelInputUtils::GetKernelInput(size_t idx) {
  if (IsNoneInput(idx)) {
    return c10::nullopt;
  }
  return input_info_.GetIntInput(idx);
}

template <>
inline float KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetFloatInput(idx);
}

template <>
inline c10::optional<float> KernelInputUtils::GetKernelInput(size_t idx) {
  if (IsNoneInput(idx)) {
    return c10::nullopt;
  }
  return input_info_.GetFloatInput(idx);
}

template <>
inline std::string KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetStrInput(idx);
}

template <>
inline std::vector<int64_t> KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetIntVecInput(idx);
}

template <>
inline std::vector<float> KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetFloatVecInput(idx);
}

template <>
inline std::vector<std::vector<int64_t>> KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetInt2DVecInput(idx);
}

template <>
inline std::vector<std::vector<float>> KernelInputUtils::GetKernelInput(size_t idx) {
  return input_info_.GetFloat2DVecInput(idx);
}

template <>
inline at::Scalar KernelInputUtils::GetKernelInput(size_t idx) {
  auto input_dtype = static_cast<TypeId>(input_info_.GetInputTypeId(idx));
  switch (input_dtype) {
    case kNumberTypeBool:
      return at::Scalar(input_info_.GetBoolInput(idx));
    case kNumberTypeInt64:
      return at::Scalar(input_info_.GetIntInput(idx));
    case kNumberTypeFloat32:
      return at::Scalar(input_info_.GetFloatInput(idx));
    default:
      throw std::runtime_error("Convert MS Scalar to at::Scalar error, unsupported Scalar type enum num:" +
        std::to_string(input_dtype) + ".");
  }
}

template <>
inline c10::optional<at::Scalar> KernelInputUtils::GetKernelInput(size_t idx) {
  if (IsNoneInput(idx)) {
    return c10::nullopt;
  }
  return GetKernelInput<at::Scalar>(idx);
}

template <>
inline at::ScalarType KernelInputUtils::GetKernelInput(size_t idx) {
  auto dtype_value = GetKernelInput<int64_t>(idx);
  switch (dtype_value) {
    case kNumberTypeBool:
      return at::ScalarType::Bool;
    case kNumberTypeInt:
      return at::ScalarType::Int;
    case kNumberTypeInt8:
      return at::ScalarType::Char;
    case kNumberTypeInt16:
      return at::ScalarType::Short;
    case kNumberTypeInt32:
      return at::ScalarType::Int;
    case kNumberTypeInt64:
      return at::ScalarType::Long;
    case kNumberTypeUInt8:
      return at::ScalarType::Byte;
    case kNumberTypeFloat:
      return at::ScalarType::Float;
    case kNumberTypeFloat16:
      return at::ScalarType::Half;
    case kNumberTypeFloat32:
      return at::ScalarType::Float;
    case kNumberTypeFloat64:
      return at::ScalarType::Double;
    case kNumberTypeBFloat16:
      return at::ScalarType::BFloat16;
    case kNumberTypeDouble:
      return at::ScalarType::Double;
    case kNumberTypeComplex:
      return at::ScalarType::ComplexFloat;
    case kNumberTypeComplex64:
      return at::ScalarType::ComplexFloat;
    case kNumberTypeComplex128:
      return at::ScalarType::ComplexDouble;
    default:
      throw std::runtime_error("Convert MS dtype to at::ScalarType error, unsupported dtype enum num:" +
        std::to_string(dtype_value) + ".");
  }
}

template <>
inline c10::optional<at::ScalarType> KernelInputUtils::GetKernelInput(size_t idx) {
  if (IsNoneInput(idx)) {
    return c10::nullopt;
  }
  return GetKernelInput<at::ScalarType>(idx);
}
}  // namespace op_plugin
#endif  //  MS_PLUGIN_OP_UTILS_H_
