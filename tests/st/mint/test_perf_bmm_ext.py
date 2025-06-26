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
import mindspore
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, ops, mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import TEST_OP, BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import time
import pytest


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def bmm_ext_forward_perf(input, mat2):
    op = mint.bmm

    for _ in range(1000):
        output = op(input, mat2)

    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        output = op(input, mat2)
    _pynative_executor.sync()
    end = time.time()

    print(f"MindSpore {op} e2e time: ", (end-start))
    return  end-start


def generate_expect_forward_perf(input, mat2):

    op = torch.bmm

    for _ in range(1000):
        op(input, mat2)

    start = time.time()
    for _ in range(1000):
        op(input, mat2)
    end = time.time()

    print(f"Torch {op} e2e time: ", end-start)
    return end-start


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_bmm_ext_perf(mode):
    shape1 = (50, 10, 20)
    shape2 = (50, 20, 10)
    input = generate_random_input(shape1, np.float32)
    mat2 = generate_random_input(shape2, np.float32)
    ms_perf = bmm_ext_forward_perf(ms.Tensor(input), ms.Tensor(mat2))
    expect_perf = generate_expect_forward_perf(torch.Tensor(input), torch.Tensor(mat2))
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

