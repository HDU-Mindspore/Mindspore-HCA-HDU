echo "build op_plugin.so"
PYTORCHPATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
PYTHONINCLUDEPATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

# 检查build目录是否存在
if [ ! -d "build" ]; then
    # 如果不存在则创建
    mkdir -p build
    echo "create build"
else
    echo "build exits"
fi

if [ -z "$CXX" ]; then
    # 如果 CC 未定义，则使用 g++
    COMPILER="g++"
else
    # 如果 CC 已定义，则使用其值
    COMPILER="$CXX"
fi

$COMPILER -MMD -MF ./build/op_plugin.o.d -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall  -fPIC -I$PYTORCHPATH/include -I$PYTORCHPATH/include/torch/csrc/api/include -I$PYTORCHPATH/include/TH -I$PYTORCHPATH/include/THC -I$PYTHONINCLUDEPATH -c ./src/op_plugin.cpp -o ./build/op_plugin.o -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

$COMPILER -MMD -MF ./build/ms_ext.o.d -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall  -fPIC -I$PYTORCHPATH/include -I$PYTORCHPATH/include/torch/csrc/api/include -I$PYTORCHPATH/include/TH -I$PYTORCHPATH/include/THC -I$PYTHONINCLUDEPATH -c ./src/ms_ext.cpp -o ./build/ms_ext.o -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17

$COMPILER -shared ./build/ms_ext.o ./build/op_plugin.o -L$PYTORCHPATH/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -o build/op_plugin.so