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
""" div op test case """
# pylint: disable=unused-variable
# pylint: disable=W0622
import random
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_scalar_input():
    return random.random()


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(input, other, rounding_mode=None):
    if rounding_mode is None:
        return torch.div(input, other)
    return torch.div(input, other, rounding_mode=rounding_mode)


def generate_expect_backward_output(input, other, grad, rounding_mode=None):
    input.requires_grad = True
    if rounding_mode is None:
        out = torch.div(input, other)
    out = torch.div(input, other, rounding_mode=rounding_mode)
    out.backward(grad)
    dx = input.grad
    return dx


def div_forward_func(input, other, rounding_mode=None):
    if rounding_mode is None:
        return mint.div(input, other)
    return mint.div(input, other, rounding_mode=rounding_mode)


def div_backward_func(input, other, rounding_mode=None):
    if rounding_mode is None:
        return ops.grad(div_forward_func, (0,))(input, other)
    return ops.grad(div_forward_func, (0,))(input, other, rounding_mode=rounding_mode)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_div_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function div.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(other), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = div_forward_func(ms.Tensor(x), ms.Tensor(other))
        output_grad = div_backward_func(ms.Tensor(x), ms.Tensor(other))
    else:
        output = jit(div_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))
        output_grad = jit(div_backward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), ms.Tensor(other))

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_divs_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function div.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_scalar_input()
    expect = generate_expect_forward_output(torch.Tensor(x), other)

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), other, torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = div_forward_func(ms.Tensor(x), other)
        output_grad = div_backward_func(ms.Tensor(x), other)
    else:
        output = jit(div_forward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), other)
        output_grad = jit(div_backward_func, backend="ms_backend", jit_level="O0")(ms.Tensor(x), other)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_divmod_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function div.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other), rounding_mode="floor")

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(other), torch.Tensor(grad),
                                                  rounding_mode="floor")

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = div_forward_func(ms.Tensor(x), ms.Tensor(other), rounding_mode="floor")
        output_grad = div_backward_func(ms.Tensor(x), ms.Tensor(other), rounding_mode="floor")
    else:
        output = jit(div_forward_func, backend="ms_backend", jit_level="O0")(
            ms.Tensor(x), ms.Tensor(other), rounding_mode="floor")
        output_grad = jit(div_backward_func, backend="ms_backend", jit_level="O0")(
            ms.Tensor(x), ms.Tensor(other), rounding_mode="floor")

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_divmods_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function div.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_scalar_input()
    expect = generate_expect_forward_output(torch.Tensor(x), other, rounding_mode="floor")

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), other, torch.Tensor(grad), rounding_mode="floor")

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = div_forward_func(ms.Tensor(x), other, rounding_mode="floor")
        output_grad = div_backward_func(ms.Tensor(x), other, rounding_mode="floor")
    else:
        output = jit(div_forward_func, backend="ms_backend", jit_level="O0")(
            ms.Tensor(x), other, rounding_mode="floor")
        output_grad = jit(div_backward_func, backend="ms_backend", jit_level="O0")(
            ms.Tensor(x), other, rounding_mode="floor")

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
