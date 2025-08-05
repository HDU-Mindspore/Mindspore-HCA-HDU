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

setlocal enabledelayedexpansion

set "BASEPATH=%~dp0"
set "OUTPUT_PATH=%BASEPATH%output\"
echo BASEPATH=%BASEPATH%
echo OUTPUT_PATH=%OUTPUT_PATH%

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    exit /b 1
)
set "PYTHON=python"


if not exist "%OUTPUT_PATH%\" mkdir "%OUTPUT_PATH%"


echo ========== Package Start ==========
call :write_version
pushd "%BASEPATH%"

%PYTHON% setup.py sdist bdist_wheel

for %%f in ("dist\ms_op_plugin-*.whl") do (
    for /f "tokens=1,2 delims=-" %%a in ("%%~nf") do (
        set "prefix=%%a-%%b"
    )
    

    %PYTHON% -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" > pyver.tmp
    set /p PY_TAGS=<pyver.tmp
    del pyver.tmp
    
    %PYTHON% -c "import platform; print('win_amd64' if platform.architecture()[0]=='64bit' else 'win32')" > arch.tmp
    set /p ARCH=<arch.tmp
    del arch.tmp

    set "new_file=!prefix!-!PY_TAGS!-none-!ARCH!.whl"
    
    ren "%%f" "!new_file!"
    move "dist\!new_file!" "%OUTPUT_PATH%" >nul && (
        echo [SUCCESS] Rename file successfully: !new_file!
    ) || (
        echo [ERROR] Move file failed: !new_file! && exit /b 1
    )
)
echo ========== Package End ==========

call :write_checksum
echo ------Successfully created mindspore_op_plugin package------
endlocal


:write_version
if not exist "%BASEPATH%version.txt" (
    for /f "tokens=*" %%b in ('git branch --show-current 2^>nul') do set "version=%%b"
    if "!version!"=="" set "version=master"
    echo !version! > "%BASEPATH%version.txt"
)
exit /b

:write_checksum
pushd "%OUTPUT_PATH%"
for %%f in (ms_op_plugin-*.whl) do (
    certutil -hashfile "%%f" SHA256 | find /v ":" | find /v "CertUtil" > "%%f.sha256"
    set /p sha256=<"%%f.sha256"
    echo %%f > "%%f.sha256"
    echo !sha256! >> "%%f.sha256"
)
popd
exit /b