#include <string.h>
#include <torch/extension.h>  // 头文件引用部分
#include <iostream>

#include "utils/ms_ext.h"

extern "C" int StackExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                        void *extra_void) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);

  auto at_output = tensors[nparam - 1];

  tensors.resize(nparam - 2);

  KernelInputInfo *kernel_input_info = static_cast<KernelInputInfo *>(extra_void);
  int64_t dim = kernel_input_info->GetKernelInput<int64_t>(nparam - 2);

  at::stack_out(at_output, tensors, dim);
  return 0;
}

extern "C" int UnstackExtView(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                              void *stream, void *extra_void){
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);

  auto input_tensor = tensors[0];

  tensors.erase(tensors.begin(), tensors.begin() + 2);

  KernelInputInfo *kernel_input_info = static_cast<KernelInputInfo *>(extra_void);
  int64_t dim = kernel_input_info->GetKernelInput<int64_t>(1);

  at::stack_out(tensors, input_tensor, dim);
  return 0;
}