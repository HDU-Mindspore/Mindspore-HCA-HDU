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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.utils.test_op_utils import TEST_OP
from tests.utils.mark_utils import arg_mark
import torch


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return torch.logical_and(x, other)


def logical_and_forward_func(x, other):
    return mint.logical_and(x, other)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_logical_and_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function logical_and.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    other = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(other))

    ms_x = ms.Tensor(x)
    ms_other = ms.Tensor(other)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = logical_and_forward_func(ms_x, ms_other)
    else:
        output = (jit(logical_and_forward_func, backend="ms_backend", jit_level="O0"))(ms_x, ms_other)

    assert np.allclose(output.asnumpy(), expect.detach().numpy(), equal_nan=True)
