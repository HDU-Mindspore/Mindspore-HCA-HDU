#!/bin/bash
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

export MS_OP_PLUGIN_PATH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${MS_OP_PLUGIN_PATH_DIR}/build"
OUTPUT_PATH="${MS_OP_PLUGIN_PATH_DIR}/output"

mk_new_dir()
{
    local create_dir="$1"

    if [[ -d "${create_dir}" ]]; then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum_tar()
{
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls lib*.tar.gz) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

check_binary_file()
{
  local binary_dir="$1"
  for cur_file in `ls "${binary_dir}"/*.o`
  do
    file_lines=`cat "${cur_file}" | wc -l`
    if [ ${file_lines} -eq 3 ]; then
        check_sha=`cat ${cur_file} | grep "oid sha256"`
        if [ $? -eq 0 ]; then
            echo "-- Warning: ${cur_file} is not a valid binary file."
            return 1
        fi
    fi
  done
  return 0
}

# Parse arguments
THREAD_NUM=32

# Create directories
mkdir -pv "${BUILD_DIR}"
mkdir -pv "${OUTPUT_PATH}"

echo "---------------- MindSpore_Op_Plugin: build start ----------------"

# Build target
cd $BUILD_DIR
cmake .. 
make -j$THREAD_NUM

if [ ! -f "libms_op_plugin.so" ];then
  echo "[ERROR] libms_op_plugin.so not exist!"
  exit 1
fi

# Copy target to output/ directory
cp libms_op_plugin.so ${OUTPUT_PATH}
cd ${OUTPUT_PATH}
tar czvf libms_op_plugin.tar.gz libms_op_plugin.so
rm -rf libms_op_plugin.so
write_checksum_tar
bash ${MS_OP_PLUGIN_PATH_DIR}/package.sh

cd -
echo "---------------- MindSpore_Op_Plugin: build end ----------------"
