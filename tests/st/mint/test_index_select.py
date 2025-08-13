# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=unused-variable
""" index_select op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-10, 10, shape).astype(dtype)

def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)

def generate_expect_forward_output(x, dim, index):
    return torch.index_select(x, dim, index)


def generate_expect_backward_output(x, dim, index, grad):
    x.requires_grad = True
    out = torch.index_select(x, dim, index)
    out.backward(grad)
    dx = x.grad
    return dx


def index_select_forward_func(x, dim, index):
    return mint.index_select(x, dim, index)


def index_select_backward_func(x, dim, index):
    return ops.grad(index_select_forward_func, (0,))(x, dim, index)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_index_select_std(mode):
    """
    Feature: pyboost function.
    Description: test function index_select.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1
    index = [0, 2]
    expect = generate_expect_forward_output(torch.Tensor(x), dim, torch.tensor(index, dtype=torch.int32))
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), dim,
                                                  torch.tensor(index, dtype=torch.int32), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = index_select_forward_func(ms.Tensor(x), dim, ms.Tensor(index, ms.int32))
        # output_grad = index_select_backward_func(ms.Tensor(x), dim, ms.Tensor(index, ms.int32))
    else:
        output = jit(index_select_forward_func, backend="ms_backend", jit_level="O0")(
            ms.Tensor(x), dim, ms.Tensor(index, ms.int32))
        # output_grad = (jit(index_select_backward_func, backend="ms_backend", jit_level="O0"))(
        #    ms.Tensor(x), dim, ms.Tensor(index, ms.int32))

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    # FIXME:index_selec算子，反向使用了原地更新算子，但更新后的Tensor没有返回到Host侧，导致反向结果错误，待定位。
    # allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
