@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==============================================
echo SVC 项目打包脚本 (Windows)
echo 目标：包含完整 Python 环境和依赖
echo ==============================================

:: 设置变量
set "PROJECT_NAME=svc_package"
set "CONDA_ENV=py310"
set "PYTHON_PATH=d:\soft\anaconda3\envs\%CONDA_ENV%"
set "TARGET_DIR=%cd%\%PROJECT_NAME%"

:: 检查 Python 环境是否存在
if not exist "%PYTHON_PATH%" (
    echo 错误：Anaconda 环境 %CONDA_ENV% 不存在！
    echo 请检查路径：%PYTHON_PATH%
    pause
    exit /b 1
)

:: 创建目标目录
if exist "%TARGET_DIR%" (
    echo 注意：目标目录 %TARGET_DIR% 已存在，将被删除...
    rmdir /s /q "%TARGET_DIR%"
)
mkdir "%TARGET_DIR%"

:: 创建临时排除文件
set "EXCLUDE_FILE=%temp%\svc_exclude_list.txt"
(
echo *.pyc
echo __pycache__
echo *.log
echo *.tmp
echo %PROJECT_NAME%
echo pack_svc.bat
echo .git
echo .idea
echo .vscode
echo data\*.wav
echo data\*.mp3
echo data\*.flac
) > "%EXCLUDE_FILE%"

:: 复制 Python 环境
echo 正在复制 Python 环境到 %TARGET_DIR%\python...
xcopy "%PYTHON_PATH%" "%TARGET_DIR%\python" /e /i /h /y /exclude:"%EXCLUDE_FILE%"

:: 复制项目文件
echo 正在复制项目文件到 %TARGET_DIR%\svc...
mkdir "%TARGET_DIR%\svc"
xcopy "%cd%" "%TARGET_DIR%\svc" /e /i /h /y /exclude:"%EXCLUDE_FILE%"

:: 创建启动脚本
echo 正在创建启动脚本...
set "START_SCRIPT=%TARGET_DIR%\run_svc.bat"
(
echo @echo off
echo chcp 65001 >nul
echo set "PYTHON_HOME=%%~dp0python"
echo set "PATH=%%PYTHON_HOME%%;%%PYTHON_HOME%%\Scripts;%%PATH%%"
echo cd /d "%%~dp0svc"
echo echo ==============================================
echo echo SVC 环境已激活！
echo echo 当前 Python 版本：
echo python --version
echo echo 当前 pip 版本：
echo pip --version
echo echo ==============================================
echo echo 可以运行以下命令：
echo echo 1. 推理：python inference.py
echo echo 2. 服务：python app.py
echo echo 3. ONNX 推理：python inference_onnx.py
echo echo ==============================================
echo cmd /k
) > "%START_SCRIPT%"

:: 创建使用说明
echo 正在创建使用说明...
set "README_FILE=%TARGET_DIR%\README_PACKAGE.md"
(
echo # SVC 打包说明
echo
echo ## 包含内容
echo - Python 3.10 完整环境 (包含 torch 2.4.1, onnx 1.20.0 等所有依赖)
echo - SVC 项目代码和模型
echo - 一键启动脚本
echo
echo ## 使用方法
echo 1. 将整个目录复制到目标 Windows PC
echo 2. 双击 `run_svc.bat` 启动环境
echo 3. 在打开的命令行中执行相应的命令
echo
echo ## 支持的命令
echo - `python inference.py` - 基本推理
echo - `python app.py` - 启动 Web 服务
echo - `python inference_onnx.py` - ONNX 推理
echo - `python real-time-gui.py` - 实时 GUI
echo
echo ## 目标平台
echo - Windows 10/11 (AMD/NVIDIA 显卡)
echo - 无需额外安装任何依赖
echo - CUDA 自动检测（如系统有 NVIDIA 显卡）
) > "%README_FILE%"

:: 清理临时文件
del "%EXCLUDE_FILE%"

echo ==============================================
echo 打包完成！
echo 打包目录：%TARGET_DIR%
echo ==============================================

pause
