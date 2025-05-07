#include <string.h>
#include <torch/extension.h>  // 头文件引用部分
#include "ms_ext.h"
#include <iostream>

extern "C" int AcosExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                       void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[1];
  torch::acos_out(at_output, at_input);
  return 0;
}

extern "C" int AtanExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                       void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[1];
  torch::atan_out(at_output, at_input);
  return 0;
}

extern "C" int InplaceReLU(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[1];
  torch::relu_out(at_output, at_input);
  return 0;
}

extern "C" int ZerosLikeExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[1];
  torch::zeros_like_out(at_output, at_input);
  return 0;
}

extern "C" int StackExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                        void *extra_void) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);

  auto at_output = tensors[nparam - 1];

  tensors.resize(nparam - 2);

  KernelInputInfo *kernel_input_info = static_cast<KernelInputInfo *>(extra_void);
  int64_t dim = kernel_input_info->GetKernelInput<int64_t>(2);

  torch::stack_out(at_output, tensors, dim);
  return 0;
}