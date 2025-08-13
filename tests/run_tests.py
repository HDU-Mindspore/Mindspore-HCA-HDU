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
""" run all test case """
import mindspore
import mindspore.context as context
import pytest

context.set_context(mode=context.PYNATIVE_MODE)
mindspore.set_device('CPU')

if __name__ == '__main__':
    pytest.main(['tests/st/mint/test_asin.py'])
    pytest.main(['tests/st/mint/test_acos.py'])
    pytest.main(['tests/st/mint/test_asinh.py'])
    pytest.main(['tests/st/mint/test_acosh.py'])
    pytest.main(['tests/st/mint/test_cumsum.py'])
    pytest.main(['tests/st/mint/test_copy_.py'])
    pytest.main(['tests/st/mint/test_sin.py'])
    pytest.main(['tests/st/mint/test_cos.py'])
    pytest.main(['tests/st/mint/test_atan.py'])
    # FIXME:relu_算子，由于当前框架CPU后端不支持原地更新算子的输入输出共用同一个Tensor，会导致反向精度不正确，因此不执行反向测试用例
    pytest.main(['tests/st/mint/test_relu_.py'])
    pytest.main(['tests/st/mint/test_stack.py'])
    pytest.main(['tests/st/mint/test_cat.py'])
    pytest.main(['tests/st/mint/test_clone.py'])
    pytest.main(['tests/st/mint/test_logical_and.py'])
    pytest.main(['tests/st/mint/test_logical_not.py'])
    pytest.main(['tests/st/mint/test_bmm_ext.py'])
    # FIXME:max算子，由于CPU后端反向不支持View+Inplace操作，因此暂不执行反向测试用例
    pytest.main(['tests/st/mint/test_max.py'])
    pytest.main(['tests/st/mint/test_max_dim.py'])
    pytest.main(['tests/st/mint/test_sum_ext.py'])
    pytest.main(['tests/st/mint/test_exp.py'])
    pytest.main(['tests/st/mint/test_zeros_like.py'])
    # FIXME:index_selec算子，反向使用了原地更新算子，但更新后的Tensor没有返回到Host侧，导致反向结果错误，待定位。
    pytest.main(['tests/st/mint/test_index_select.py'])
    pytest.main(['tests/st/mint/test_div.py'])
    pytest.main(['tests/st/mint/test_zeros.py'])
    pytest.main(['tests/st/mint/test_ones.py'])


    pytest.main(['tests/st/mint/test_perf_acos.py'])
    pytest.main(['tests/st/mint/test_perf_copy_.py'])
    pytest.main(['tests/st/mint/test_perf_sin.py'])
    pytest.main(['tests/st/mint/test_perf_atan.py'])
    # FIXME: relu_ 算子性能不达标，怀疑是由于框架多申请了一个输出Tensor导致。
    pytest.main(['tests/st/mint/test_perf_relu_.py'])
    pytest.main(['tests/st/mint/test_perf_stack.py'])
    pytest.main(['tests/st/mint/test_perf_cat.py'])
    pytest.main(['tests/st/mint/test_perf_clone.py'])
    # FIXME: index 算子性能不达标，原因是torch走View，MS暂时不支持。
    pytest.main(['tests/st/mint/test_perf_index.py'])
    pytest.main(['tests/st/mint/test_perf_logical_and.py'])
    pytest.main(['tests/st/mint/test_perf_logical_not.py'])
    pytest.main(['tests/st/mint/test_perf_index_select.py'])
    pytest.main(['tests/st/mint/test_perf_acosh.py'])
    pytest.main(['tests/st/mint/test_perf_asinh.py'])
    pytest.main(['tests/st/mint/test_perf_max.py'])
    pytest.main(['tests/st/mint/test_perf_max_dim.py'])
    pytest.main(['tests/st/mint/test_perf_bmm_ext.py'])
    pytest.main(['tests/st/mint/test_perf_sum_ext.py'])
    pytest.main(['tests/st/mint/test_perf_cumsum.py'])
    # FIXME: zeros_like 算子性能较差，原因未明。
    pytest.main(['tests/st/mint/test_perf_zeros_like.py'])
    pytest.main(['tests/st/mint/test_perf_div.py'])
    pytest.main(['tests/st/mint/test_perf_ones.py'])
    pytest.main(['tests/st/mint/test_perf_zeros.py'])
