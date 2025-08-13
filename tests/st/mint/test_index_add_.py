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
""" index_add_ op test case """
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


def generate_expect_forward_output(x, dim, index, source, alpha):
    x = x * 1
    x.index_add_(dim, index, source, alpha=alpha)
    return x


class InplaceReluModule(torch.nn.Module):
    def __init__(self):
        pass


    def forward(self, x, dim, index, source, alpha):
        x = x * 1
        return x.index_add_(dim, index, source, alpha=alpha)


def generate_expect_backward_output(x, dim, index, source, alpha, grad):
    x.requires_grad = True
    torch_net = InplaceReluModule()
    out = torch_net(x, dim, index, source, alpha)
    out.backward(grad)
    dx = x.grad
    return dx


def index_add__forward_func(x, dim, index, source, alpha):
    x = x * 1
    x.index_add_(dim, index, source, alpha=alpha)
    return x


def index_add__backward_func(x, dim, index, source, alpha):
    return ops.grad(index_add__forward_func, (0,))(x, dim, index, source, alpha)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_index_add__std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function index_add_.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    source = generate_random_input((2, 2, 4), np.float32)
    dim = 1
    index = [0, 2]
    alpha = 2
    expect = generate_expect_forward_output(torch.Tensor(x), dim, torch.tensor(index, dtype=torch.int32),
                                            torch.Tensor(source), alpha)

    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    # expect_grad = generate_expect_backward_output(torch.Tensor(x), dim, torch.tensor(index, dtype=torch.int32),
    #    torch.Tensor(source), alpha, torch.Tensor(grad))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = index_add__forward_func(ms.Tensor(x), dim, ms.tensor(index, dtype=ms.int32), ms.Tensor(source), alpha)
        # output_grad = index_add__backward_func(ms.Tensor(x), dim,
        #    ms.tensor(index, dtype=ms.int32), ms.Tensor(source), alpha)
    else:
        output = (jit(index_add__forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(x), dim, ms.tensor(index, dtype=ms.int32), ms.Tensor(source), alpha)
        # output_grad = (jit(index_add__backward_func, backend="ms_backend", jit_level="O0"))(
        #    ms.Tensor(x), dim, ms.tensor(index, dtype=ms.int32), ms.Tensor(source), alpha)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    # allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
