# Copyright 2024 Huawei Technologies Co., Ltd
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
""" sum op test case """
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(x, dim):
    return torch.sum(x, dim)


def generate_expect_backward_output(x, dim, grad):
    x.requires_grad = True
    out = torch.sum(x, dim)
    out.backward(grad)
    dx = x.grad
    return dx


def sum_forward_func(x, dim):
    return mint.sum(x, dim)


def sum_backward_func(x, dim):
    return ops.grad(sum_forward_func, (0,))(x, dim)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_sum_ext_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function sum_ext.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    dim = (0, 1)
    expect = generate_expect_forward_output(torch.Tensor(x), dim)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), dim, torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = sum_forward_func(ms.Tensor(x), dim)
        output_grad = sum_backward_func(ms.Tensor(x), dim)
    else:
        output = (jit(sum_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim)
        output_grad = (jit(sum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
