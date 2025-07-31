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
""" logical_not op test case """
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape):
    return np.random.choice([False, True], shape, [0.5, 0.5])


def generate_expect_forward_output(x):
    return torch.logical_not(x)


def logical_not_forward_func(x):
    return mint.logical_not(x)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_logical_not_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function logical_not.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4))
    expect = generate_expect_forward_output(torch.Tensor(x))

    ms_x = ms.Tensor(x)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = logical_not_forward_func(ms_x)
    else:
        output = (jit(logical_not_forward_func, backend="ms_backend", jit_level="O0"))(ms_x)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
