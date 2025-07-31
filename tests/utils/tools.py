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
""" test tools"""
import numpy as np

def allclose_nparray(data_expected, data_ms, rtol=0, atol=0, equal_nan=True):
    assert data_expected.dtype == data_ms.dtype
    assert data_expected.shape == data_ms.shape
    np.testing.assert_allclose(data_expected, data_ms, rtol=rtol, atol=atol, equal_nan=equal_nan)
