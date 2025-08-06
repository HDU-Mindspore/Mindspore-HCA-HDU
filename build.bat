@echo off
REM Copyright 2025 Huawei Technologies Co., Ltd
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.

REM This should run in "x64 Native Tools Command Prompt for VS 2022"

setlocal enabledelayedexpansion


set "MS_OP_PLUGIN_PATH_DIR=%~dp0"
set "BUILD_DIR=%MS_OP_PLUGIN_PATH_DIR%build\"
set "OUTPUT_PATH=%MS_OP_PLUGIN_PATH_DIR%output\"
echo MS_OP_PLUGIN_PATH_DIR:%MS_OP_PLUGIN_PATH_DIR%
echo BUILD_DIR:%BUILD_DIR%
echo OUTPUT_PATH:%OUTPUT_PATH%


call :mk_new_dir "%BUILD_DIR%"
echo Created %BUILD_DIR%
call :mk_new_dir "%OUTPUT_PATH%"
echo Created %OUTPUT_PATH%

echo -------------------- MindSpore_Op_Plugin: build start --------------------


cmake -G "Visual Studio 17 2022" -B "build" -S "." 
cmake --build build --config Release
copy "third_party\libtorch\lib\x86_64\win\*.dll" "build\Release\" || (
    echo [ERROR] DLL copy failed! && exit /b 1
)


pushd "%BUILD_DIR%"
if not exist "Release\ms_op_plugin.dll" (
    echo [ERROR] ms_op_plugin.dll not exist! && popd && exit /b 1
)
copy "Release\ms_op_plugin.dll" "%OUTPUT_PATH%" >nul || (
    echo [ERROR] Failed to copy DLL to output && popd && exit /b 1
)
popd


pushd "%OUTPUT_PATH%"
tar -czf ms_op_plugin.tar.gz ms_op_plugin.dll || (
    echo [ERROR] TAR creation failed! && popd && exit /b 1
)
del ms_op_plugin.dll >nul
call :write_checksum_tar
call "%MS_OP_PLUGIN_PATH_DIR%package.bat" || (
    echo [ERROR] package.bat failed! && popd && exit /b 1
)
popd

echo -------------------- MindSpore_Op_Plugin: build end --------------------
endlocal
exit /b 0  




:mk_new_dir
set "create_dir=%~1"
if exist "!create_dir!\" (
    rmdir /s /q "!create_dir!" >nul 2>&1 || exit /b 1
)
mkdir "!create_dir!" >nul 2>&1 || (
    echo [ERROR] Cannot create dir: !create_dir! && exit /b 1
)
exit /b 0 

:write_checksum_tar
for %%f in (*.tar.gz) do (
    certutil -hashfile "%%f" SHA256 | find /v ":" > "%%f.sha256" || (
        echo [ERROR] Checksum failed for %%f && exit /b 1
    )
    set /p sha256=<"%%f.sha256"
    >"%%f.sha256" (
        echo %%f
        echo !sha256!
    )
)
exit /b 0

:check_binary_file
set "binary_dir=%~1"
for %%f in ("%binary_dir%\*.o") do (
    find /c /v "" < "%%f" > lines.tmp || exit /b 1
    set /p file_lines=<lines.tmp
    if !file_lines! equ 3 (
        find "oid sha256" "%%f" >nul && (
            echo -- Warning: %%f is not a valid binary file.
            del lines.tmp >nul
            exit /b 1
        )
    )
)
del lines.tmp >nul
exit /b 0