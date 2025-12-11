@echo off

set "BAT_DIR=%~dp0"
echo %BAT_DIR%
if "%BAT_DIR:~-1%"=="\" set "BAT_DIR=%BAT_DIR:~0,-1%"
echo %BAT_DIR%

set "PYTHON_DIR=%BAT_DIR%\python"
echo %PYTHON_DIR%

set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
echo %PYTHON_EXE%

if not exist "%PYTHON_EXE%" (
    echo python not exists
    pause
    exit /b 1
)

set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
set "PROJECT_DIR=%BAT_DIR%\svc"

if not exist "%PROJECT_DIR%" (
    echo [ERROR dir] not exists:"%PROJECT_DIR%"
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"
echo ===== SVC evn activate =====
"%PYTHON_EXE%" --version
"%PYTHON_EXE%" -m pip --version
echo ==========================

"%PYTHON_EXE%" svc_backend.py

cd /d "%BAT_DIR%"
cmd /k