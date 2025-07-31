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
""" index op test case """
import pytest
import numpy as np
import mindspore as ms
from mindspore.ops.auto_generate.gen_ops_def import index
from tests.utils import test_utils
from tests.utils.mark_utils import arg_mark
from tests.utils.tools import allclose_nparray
import torch


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def index_forward_func(x, indices):
    return index(x, indices)


@test_utils.run_with_cell
def index_backward_func(x, indices):
    return ms.grad(index_forward_func, (0,))(x, indices)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_index_std(mode):
    """
    Feature: pyboost function.
    Description: test function index forward.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    x = generate_random_input((3, 4, 5, 6, 7), np.float64)

    # shape(0,) and shape(0,0,0,0,0,0,0,0,0)
    indices1 = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    indices2 = np.array([0, 1, 2], dtype=np.int32)
    indices3 = np.array(([1], [1]), dtype=np.int32)

    ms_indices1 = ms.tensor(indices1)
    ms_indices2 = ms.tensor(indices2)
    ms_indices3 = ms.tensor(indices3)

    output_1 = index_forward_func(ms.tensor(x), [ms_indices1, ms_indices2])
    output_2 = index_forward_func(ms.tensor(x), [ms_indices3, ms_indices3, ms_indices2, ms_indices2, ms_indices1])

    pt_indices1 = torch.tensor(indices1)
    pt_indices2 = torch.tensor(indices2)
    pt_indices3 = torch.tensor(indices3)

    expect_1 = torch.tensor(x)[pt_indices1, pt_indices2]
    expect_2 = torch.tensor(x)[pt_indices3, pt_indices3, pt_indices2, pt_indices2, pt_indices1]

    allclose_nparray(expect_1.numpy(), output_1.asnumpy())
    allclose_nparray(expect_2.numpy(), output_2.asnumpy())
