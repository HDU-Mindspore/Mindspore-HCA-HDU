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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.test_op_utils import TEST_OP
from tests.utils.mark_utils import arg_mark
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_ones_grad(shape, dtype):
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(input, mat2):
    return torch.bmm(input, mat2)


def generate_expect_backward_output(input, mat2, grad):
    input.requires_grad = True
    out = torch.bmm(input, mat2)
    out.backward(grad)
    dx = input.grad
    return dx


def bmm_forward_func(input, mat2):
    return mint.bmm(input, mat2)


def bmm_backward_func(input, mat2):
    return ops.grad(bmm_forward_func, (0,))(input, mat2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_bmm_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function bmm.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    mat2 = generate_random_input((2, 4, 5), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(mat2))

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(mat2), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = bmm_forward_func(ms.Tensor(x), ms.Tensor(mat2))
        output_grad = bmm_backward_func(ms.Tensor(x), ms.Tensor(mat2))
    else:
        output = (jit(bmm_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(mat2))
        output_grad = (jit(bmm_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(mat2))

    assert np.allclose(output.asnumpy(), expect.detach().numpy(), equal_nan=True)
    assert np.allclose(output_grad.asnumpy(), expect_grad.detach().numpy(), equal_nan=True)


