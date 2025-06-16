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
from mindspore.common import mutable
from mindspore import Tensor, mint
from tests.utils import test_utils
from tests.utils.test_op_utils import TEST_OP
from tests.utils.mark_utils import arg_mark
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-10, 10, shape).astype(dtype)


def generate_expect_forward_output(x, dim, index):
    return torch.index_select(x, dim, index)


def index_select_forward_func(x, dim, index):
    return mint.index_select(x, dim, index)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_index_std(mode):
    """
    Feature: pyboost function.
    Description: test function index_select.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1

    expect = generate_expect_forward_output(torch.Tensor(x), dim, torch.tensor([0, 2], dtype=torch.int32))

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = index_select_forward_func(ms.Tensor(x), dim, ms.Tensor([0, 2], ms.int32))
    else:
        output = (jit(index_select_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim, ms.Tensor([0, 2], ms.int32))

    assert np.allclose(output.asnumpy(), expect.detach().numpy(), equal_nan=True)
