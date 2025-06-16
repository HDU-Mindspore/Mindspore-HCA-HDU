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
    return np.random.randn(*shape).astype(dtype)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(x, dim, dtype=None):
    return torch.cumsum(x, dim, dtype=dtype)


def generate_expect_backward_output(x, grad, dim, dtype=None):
    x.requires_grad = True
    out = torch.cumsum(x, dim, dtype=dtype)
    out.backward(grad)
    dx = x.grad
    return dx


def cumsum_forward_func(x, dim, dtype=None):
    return mint.cumsum(x, dim, dtype=dtype)


def cumsum_backward_func(x, dim, dtype=None):
    return ops.grad(cumsum_forward_func, (0,))(x, dim, dtype=dtype)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_cumsum_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function cumsum.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    dim = np.random.randint(0, 3, dtype=np.int64)

    expect = generate_expect_forward_output(torch.Tensor(x), dim)

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(grad), dim)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_forward_func(ms.Tensor(x), dim.item())
        output_grad = cumsum_backward_func(ms.Tensor(x), dim.item())
    else:
        output = (jit(cumsum_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim.item())
        output_grad = (jit(cumsum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim.item())

    assert np.allclose(output.asnumpy(), expect.detach().numpy(), equal_nan=True)
    assert np.allclose(output_grad.asnumpy(), expect_grad.detach().numpy(), equal_nan=True)
