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

import os
import platform

from setuptools import setup

pwd = os.path.dirname(os.path.realpath(__file__))

def _read_file(filename):
    with open(os.path.join(pwd, filename), encoding='UTF-8') as f:
        return f.read()


version = _read_file('version.txt').replace("\n", "")


arch = platform.machine().replace("AMD64", "x86_64")
if arch =='x86_64':
    os_name = 'win' if os.name=='nt' else 'linux'
    arch = os.path.join(arch, os_name)
    
torch_lib = os.path.join('third_party', 'libtorch', 'lib', arch)
if "win" not in arch:
    package_data = {
        'ms_op_plugin': [
            '*.so*',
        ],
        'libtorch' : [
            '*.so',
        ]
    }
    package_dir={'ms_op_plugin': 'build', 'libtorch': torch_lib}
else:
    package_data = {
        'ms_op_plugin': [
            '*',
        ]
    }
    package_dir={'ms_op_plugin': 'build\\Release'}
print(torch_lib)

setup(
    name='ms_op_plugin',
    author='The MindSpore op_plugin Authors',
    author_email='contact@mindspore.cn',
    download_url='https://gitee.com/mindspore/mindspore_op_plugin/tags',
    project_urls={
        'Sources': 'https://gitee.com/mindspore/mindspore_op_plugin',
        'Issue Tracker': 'https://gitee.com/mindspore/mindspore_op_plugin/issues',
    },
    description="An op_plugin for MindSpore",
    license='Apache 2.0',
    package_data=package_data,
    package_dir=package_dir,
    include_package_data=True,
    classifiers=[
        'License :: OSI Approved :: Apache Software License'
    ]
)
