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
""" zeros op test case """
# pylint: disable=unused-variable
import pytest
import mindspore as ms
from mindspore import mint, jit
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(size):
    return size


def generate_expect_forward_output(size, dtype):
    return torch.zeros(size, dtype=dtype)


def zeros_forward_func(size, dtype):
    return mint.zeros(size, dtype=dtype)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_zeros_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function zeros.
    Expectation: expect correct result.
    """
    size = (2, 3, 4)
    expect = generate_expect_forward_output(size, torch.float16)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = zeros_forward_func(size, ms.float16)
    else:
        output = (jit(zeros_forward_func, backend="ms_backend", jit_level="O0"))(size, ms.float16)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
