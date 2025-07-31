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
""" copy_ op test case """
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(dst, src):
    dst = dst * 1
    dst.copy_(src)
    return dst


def generate_expect_backward_output(dst, src, grad):
    src.requires_grad = True
    dst = dst * 1
    out = dst.copy_(src)
    out.backward(grad)
    d_src = src.grad
    return d_src


def copy__forward_func(dst, src):
    dst = dst * 1
    dst.copy_(src)
    return dst


def copy__backward_func(dst, src):
    return ops.grad(copy__forward_func, (1,))(dst, src)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_copy__std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy_.
    Expectation: expect correct result.
    """
    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(dst), torch.Tensor(src))

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(dst), torch.Tensor(src), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = copy__forward_func(ms.Tensor(dst), ms.Tensor(src))
        output_grad = copy__backward_func(ms.Tensor(dst), ms.Tensor(src))
    else:
        output = (jit(copy__forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(dst), ms.Tensor(src))
        output_grad = (jit(copy__backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(dst), ms.Tensor(src))

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
