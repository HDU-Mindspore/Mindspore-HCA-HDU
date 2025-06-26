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
import mindspore
import mindspore as ms
import mindspore.context as context
from mindspore import tensor, ops, mint
from mindspore.ops.auto_generate.gen_ops_def import index
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import TEST_OP, BACKGROUND_NOISE
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import time
import pytest


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def stack_forward_perf(input, indices):
    op = index
    print("================shape: ", input.shape)

    for _ in range(1000):
        output = op(input, indices)

    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        output = op(input, indices)
    _pynative_executor.sync()
    end = time.time()

    print(f"MindSpore {op} e2e time: ", (end-start))
    return  end-start


def generate_expect_forward_perf(input, indices):

    print("================shape: ", input.shape)

    for _ in range(1000):
        input[indices]

    start = time.time()
    for _ in range(1000):
        input[indices]
    end = time.time()

    print(f"Torch index e2e time: ", end-start)
    return end-start


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_stack_perf(mode):
    shape = (10, 10, 10, 10, 10, 10, 5)
    input = generate_random_input(shape, np.float32)
    indices1 = np.random.randint(0, 10, (10, 10), dtype=np.int32)
    
    ms_perf = stack_forward_perf(ms.tensor(input), [ms.tensor(indices1), ms.tensor(indices1)])
    expect_perf = generate_expect_forward_perf(torch.tensor(input), [torch.tensor(indices1), torch.tensor(indices1)])
    assert np.less(ms_perf - BACKGROUND_NOISE, expect_perf * 1.1).all()

