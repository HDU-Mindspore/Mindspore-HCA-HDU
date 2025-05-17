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

#include "utils/ms_ext.h"

extern "C" int Div(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCPU);
  auto at_input1 = tensors[0];
  auto at_input2 = tensors[1];
  auto at_output = tensors[2];
  at::div_out(at_output, at_input1, at_input2);
  return 0;
}