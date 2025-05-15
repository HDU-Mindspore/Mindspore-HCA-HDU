#include <string.h>
#include <torch/extension.h>  // 头文件引用部分
#include <iostream>

#include "utils/ms_ext.h"

extern "C" int ZerosLikeExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input = tensors[0];
  auto at_output = tensors[1];
  at::zeros_like_out(at_output, at_input);
  return 0;
}