@REM This should run in "x64 Native Tools Command Prompt for VS 2022"
cmake -G "Visual Studio 17 2022" -B build -S .
cmake --build build --config Release 
copy third_party\libtorch\lib\x86_64\win\*.dll build\Release\     