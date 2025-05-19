import mindspore
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, ops, mint
from mindspore.common.api import _pynative_executor
from tests.utils.test_op_utils import TEST_OP
from tests.utils.mark_utils import arg_mark
import torch
import numpy as np
import time
import pytest


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def acos_forward_perf(input):
    op = mint.acos
    print("================shape: ", input.shape)

    for _ in range(1000):
        output = op(input)

    _pynative_executor.sync()
    start = time.time()
    for _ in range(1000):
        output = op(input)
    _pynative_executor.sync()
    end = time.time()

    print(f"MindSpore {op} e2e time: ", (end-start))
    return  end-start


def generate_expect_forward_perf(input):

    op = torch.acos
    print("================shape: ", input.shape)

    for _ in range(1000):
        op(input)

    start = time.time()
    for _ in range(1000):
        op(input)
    end = time.time()

    print(f"Torch {op} e2e time: ", end-start)
    return end-start


@arg_mark(plat_marks=['cpu_linux'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_acos_perf(mode):
    shape = (10, 10, 10, 10, 10, 10, 10)
    input = generate_random_input(shape, np.float32)
    ms_perf = acos_forward_perf(ms.Tensor(input))
    expect_perf = generate_expect_forward_perf(torch.Tensor(input))
    assert np.less(ms_perf, expect_perf * 1.1).all()

