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
""" relu_ op test case """
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


def generate_expect_forward_output(x):
    x = x * 1
    torch.nn.functional.relu_(x)
    return x


class InplaceReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.functional.relu_


    def forward(self, x):
        x = x * 1
        return self.op(x)


def generate_expect_backward_output(x, grad):
    x.requires_grad = True
    torch_net = InplaceReluModule()
    out = torch_net(x)
    out.backward(grad)
    dx = x.grad
    return dx


def relu__forward_func(x):
    x = x * 1
    mint.nn.functional.relu_(x)
    return x


def relu__backward_func(x):
    return ops.grad(relu__forward_func, (0,))(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_relu__std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function relu_.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x))

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    # expect_grad = generate_expect_backward_output(torch.Tensor(x), torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = relu__forward_func(ms.Tensor(x))
        output_grad = relu__backward_func(ms.Tensor(x))
    else:
        output = (jit(relu__forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x))
        output_grad = (jit(relu__backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x))

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    # allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
