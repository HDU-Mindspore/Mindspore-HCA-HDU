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


import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.test_op_utils import TEST_OP
from tests.utils.mark_utils import arg_mark
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-10, 10, shape).astype(dtype)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(x):
    return torch.asinh(x)


def generate_expect_backward_output(x, grad):
    x.requires_grad = True
    out = torch.asinh(x)
    out.backward(grad)
    dx = x.grad
    return dx


def asinh_forward_func(x):
    return mint.asinh(x)


def asinh_backward_func(x):
    return ops.grad(asinh_forward_func, (0,))(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_asinh_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function asinh.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x))

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = asinh_forward_func(ms.Tensor(x))
        output_grad = asinh_backward_func(ms.Tensor(x))
    else:
        output = (jit(asinh_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x))
        output_grad = (jit(asinh_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x))

    assert np.allclose(output.asnumpy(), expect.detach().numpy(), equal_nan=True)
    assert np.allclose(output_grad.asnumpy(), expect_grad.detach().numpy(), equal_nan=True)
