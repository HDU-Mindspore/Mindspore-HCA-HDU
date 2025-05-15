#include <string.h>
#include <vector>
#include <torch/extension.h>  // 头文件引用部分

// 将 mindspore kernel 的 inputs/outputs 转换为 pytorch 的 tensor
std::vector<at::Tensor> get_torch_tensors(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                          c10::Device device);

// 将入参没有输出的pytorch 算子的计算结果拷贝到kernel的输出内存
void output_memcpy(void *output, const torch::Tensor &t);

class CustomKernelData {
 public:
  CustomKernelData() = default;
  virtual ~CustomKernelData() = default;
};

class KernelInputInfo {
 public:
  KernelInputInfo() = default;
  virtual ~KernelInputInfo() = default;

  template <typename T>
  inline T GetKernelInput(size_t idx) {
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  void SetKernelData(CustomKernelData *kernel_data) { kernel_data_ = kernel_data; }
  CustomKernelData *KernelData() const { return kernel_data_; }

  void DestructKernelData() {
    delete kernel_data_;
    kernel_data_ = nullptr;
  }
  virtual size_t GetInputSize() = 0;

 private:
  virtual bool GetBoolInput(size_t idx) = 0;
  virtual int64_t GetIntInput(size_t idx) = 0;
  virtual float GetFloatInput(size_t idx) = 0;
  virtual std::string GetStrInput(size_t idx) = 0;

  virtual std::vector<int64_t> GetIntVecInput(size_t idx) = 0;
  virtual std::vector<float> GetFloatVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<float>> GetFloat2DVecInput(size_t idx) = 0;
  std::vector<size_t> workspace_;

  CustomKernelData *kernel_data_{nullptr};
};

template <>
inline bool KernelInputInfo::GetKernelInput(size_t idx) {
  return GetBoolInput(idx);
}

template <>
inline int64_t KernelInputInfo::GetKernelInput(size_t idx) {
  return GetIntInput(idx);
}

template <>
inline float KernelInputInfo::GetKernelInput(size_t idx) {
  return GetFloatInput(idx);
}

template <>
inline std::string KernelInputInfo::GetKernelInput(size_t idx) {
  return GetStrInput(idx);
}

template <>
inline std::vector<int64_t> KernelInputInfo::GetKernelInput(size_t idx) {
  return GetIntVecInput(idx);
}

template <>
inline std::vector<float> KernelInputInfo::GetKernelInput(size_t idx) {
  return GetFloatVecInput(idx);
}

template <>
inline std::vector<std::vector<int64_t>> KernelInputInfo::GetKernelInput(size_t idx) {
  return GetInt2DVecInput(idx);
}

template <>
inline std::vector<std::vector<float>> KernelInputInfo::GetKernelInput(size_t idx) {
  return GetFloat2DVecInput(idx);
}