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

set -e

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
  for cur_file in "${binary_dir}"/*.o
  do
    file_lines=`cat "${cur_file}" | wc -l`
    if [ ${file_lines} -eq 3 ]; then
        if [ $? -eq 0 ]; then
            echo "-- Warning: ${cur_file} is not a valid binary file."
            return 1
        fi
    fi
  done
  return 0
}

usage()
{
  echo "Usage:"
  echo "bash build.sh [-a on|off] [-h] \\"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -a Enable ASAN, default off"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

build_option_proc_a()
{
  check_on_off $OPTARG a
  export ENABLE_ASAN="$OPTARG"
}

init_default_options()
{
  export THREAD_NUM=32
  export ENABLE_ASAN="off"
}

# check and set options
process_options()
{
  # Process the options
  while getopts 'a:hj:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      a)
        build_option_proc_a
        ;;
      j)
        export THREAD_NUM=$OPTARG
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
        ;;
    esac
  done
}

build_ms_plugin()
{
  export MS_OP_PLUGIN_PATH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd)"
  BUILD_DIR="${MS_OP_PLUGIN_PATH_DIR}/build"
  OUTPUT_PATH="${MS_OP_PLUGIN_PATH_DIR}/output"

  # Create directories
  mkdir -pv "${BUILD_DIR}"
  mkdir -pv "${OUTPUT_PATH}"

  CMAKE_ARGS=""
  if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
  fi
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"

  echo "---------------- MindSpore_Op_Plugin: build start ----------------"
  # Build target
  cd $BUILD_DIR
  cmake ${CMAKE_ARGS} .. 
  make -j$THREAD_NUM
}
init_default_options
process_options "$@"
build_ms_plugin

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
