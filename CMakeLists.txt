cmake_minimum_required(VERSION 3.14.0)
project(MindSpore_op_plugin)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
        message(FATAL_ERROR "GCC version must be 7.3.0 and above, but found ${CMAKE_CXX_COMPILER_VERSION}")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11.3.0)
        message(WARNING "GCC version ${CMAKE_CXX_COMPILER_VERSION} is greater than 11.3.0, may cause unknown problems.")
    endif()
endif()

# Generic compilation options
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
if(NOT SUPPORT_CXX17)
  message(
    FATAL_ERROR "-- please choose a compiler which support C++ 17 standard")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -Wall -fPIC -fstack-protector-all -Wno-uninitialized -Wno-parentheses")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pipe -Wall -Wno-unknown-pragmas -Wno-unused-result -Wsign-compare -fPIC -fstack-protector-all -Wno-uninitialized -Wno-parentheses -D_GLIBCXX_USE_CXX11_ABI -DTORCH_API_INCLUDE_EXTENSION_H")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("-- Build in Debug mode")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -rdynamic -ftrapv")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -rdynamic -ftrapv")
else()
  message("-- Build in Release mode")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -D_FORTIFY_SOURCE=2 -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -D_FORTIFY_SOURCE=2 -Werror")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set Path
set(OP_PLUGIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
get_filename_component(OP_PLUGIN_SOURCE_DIR "${OP_PLUGIN_SOURCE_DIR}" ABSOLUTE)
message("OP_PLUGIN_SOURCE_DIR = ${OP_PLUGIN_SOURCE_DIR}")


# libtorch 

set(TORCH_LIBS ${OP_PLUGIN_SOURCE_DIR}/third_party/libtorch)

include_directories(${OP_PLUGIN_SOURCE_DIR}/op_plugin)
include_directories(${TORCH_LIBS}/include)
include_directories(${TORCH_LIBS}/include/torch/csrc/api/include)
include_directories(${TORCH_LIBS}/include/TH)
include_directories(${TORCH_LIBS}/include/THC)

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TORCH_LIBS_PATH ${TORCH_LIBS}/lib/aarch64)
else(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(TORCH_LIBS_PATH ${TORCH_LIBS}/lib/x86_64)
endif()
message(STATUS "this is ${TORCH_LIBS_PATH}")

find_library(TORCH_CPU NAMES torch_cpu PATHS ${TORCH_LIBS_PATH})
find_library(C10 NAMES c10 PATHS ${TORCH_LIBS_PATH})
message("libraries： ${TORCH_CPU} ${C10}")

#set python_path
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    if(WIN32)
        if(Python3_DIR)
            message("Python3_DIR set already: " ${Python3_DIR})
        else()
            string(LENGTH ${PYTHON_LIBRARIES} PYTHON_LIBRARIES_LEN)
            string(LENGTH "libpythonxx.a" Python3_NAME_LEN)
            math(EXPR Python3_DIR_LEN  ${PYTHON_LIBRARIES_LEN}-${Python3_NAME_LEN})
            string(SUBSTRING ${Python3_LIBRARIES} 0 ${Python3_DIR_LEN} Python3_DIR)
            message("Python3_DIR: " ${Python3_DIR})
        endif()
        link_directories(${Python3_DIR})
    endif()
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})



# Source file lists
file(
  GLOB
  OP_PLUGIN_SRCS
  ${OP_PLUGIN_SOURCE_DIR}/op_plugin/ops/kernel/*.cc
  ${OP_PLUGIN_SOURCE_DIR}/op_plugin/utils/*.cc)

add_library(ms_op_plugin SHARED ${OP_PLUGIN_SRCS})

target_link_libraries(ms_op_plugin PUBLIC ${TORCH_CPU} ${C10})

install(TARGETS ms_op_plugin DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib)





