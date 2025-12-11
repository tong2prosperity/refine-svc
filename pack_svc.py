#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import fnmatch

print("==============================================")
print("SVC Project Packaging Script (Python Version)")
print("Target: Include complete Python environment and dependencies")
print("==============================================")

# Set variables
PROJECT_NAME = "svc_package"
CONDA_ENV = "svc"
PYTHON_PATH = r"D:\soft\anaconda3\envs\svc"  # Adjust according to your environment
TARGET_DIR = os.path.join(os.getcwd(), PROJECT_NAME)

# Check if Python environment exists
if not os.path.exists(PYTHON_PATH):
    print(f"Error: Anaconda environment {CONDA_ENV} does not exist!")
    print(f"Please check path: {PYTHON_PATH}")
    input("Press any key to exit...")
    sys.exit(1)

# Create/clean target directory
if os.path.exists(TARGET_DIR):
    print(f"Note: Target directory {TARGET_DIR} already exists, it will be deleted...")
    shutil.rmtree(TARGET_DIR)

os.makedirs(TARGET_DIR)
print(f"Created target directory: {TARGET_DIR}")

# Define ignore rules for files/directories
def custom_ignore(dir_path, contents):
    ignored = []
    # General ignore files
    general_ignore_files = ["*.pyc", "*.log", "*.tmp", "pack_svc*.bat", "pack_svc.py", "test_exclude.txt"]
    # General ignore directories
    general_ignore_dirs = ["__pycache__", ".git", ".idea", ".vscode", PROJECT_NAME, "svc_package"]
    # Specific ignore files for data directory
    data_ignore_files = ["*.wav", "*.mp3", "*.flac", "*.m4a"]

    for content in contents:
        full_content_path = os.path.join(dir_path, content)
        if os.path.isdir(full_content_path):
            # 忽略指定目录
            if content in general_ignore_dirs:
                ignored.append(content)
        else:
            # 忽略通用文件
            for pattern in general_ignore_files:
                if fnmatch.fnmatch(content, pattern):
                    ignored.append(content)
                    break
            # 忽略data目录下的音频文件
            if os.path.basename(dir_path) == "data":
                for pattern in data_ignore_files:
                    if fnmatch.fnmatch(content, pattern):
                        ignored.append(content)
                        break

    return ignored

try:
    # Copy Python environment
    print("Copying Python environment...")
    python_target = os.path.join(TARGET_DIR, "python")
    # First check if source Python executable exists
    source_python_exe = os.path.join(PYTHON_PATH, "python.exe")
    if not os.path.exists(source_python_exe):
        print(f"ERROR: Source Python executable not found at {source_python_exe}!")
        input("Press any key to exit...")
        sys.exit(1)
    print(f"Found source Python: {source_python_exe}")

    shutil.copytree(PYTHON_PATH, python_target, ignore=custom_ignore)

    # Verify Python was copied
    dest_python_exe = os.path.join(python_target, "python.exe")
    if not os.path.exists(dest_python_exe):
        print(f"ERROR: Python executable not copied to {dest_python_exe}!")
        input("Press any key to exit...")
        sys.exit(1)
    print(f"OK: Python environment copied to {python_target}")
    print(f"Python executable at: {dest_python_exe}")

    # Copy project files
    print("Copying project files...")
    svc_target = os.path.join(TARGET_DIR, "svc")
    shutil.copytree(".", svc_target, ignore=custom_ignore)
    print("OK: Project files copy completed")

    # Create startup script
    print("Creating startup script...")
    start_script_path = os.path.join(TARGET_DIR, "run_svc.bat")
    # Use UTF-8 encoding with BOM for full Windows compatibility
    with open(start_script_path, "wb") as f:
        f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
        content = f"""@echo off
setlocal enabledelayedexpansion

:: Set UTF-8 encoding
chcp 65001 >nul 2>&1

:: Set Python path
set "PYTHON_DIR=%%~dp0python"
set "PYTHON_EXE=%%PYTHON_DIR%%\\python.exe"

:: Verify Python exists
if not exist "%%PYTHON_EXE%%" (
    echo Error: Python interpreter not found! Please check if %%PYTHON_DIR%% contains python.exe
    pause
    exit /b 1
)

:: Set environment variables
set "PYTHON_HOME=%%PYTHON_DIR%%"
set "PATH=%%PYTHON_HOME%%;%%PYTHON_HOME%%\\Scripts;%%PATH%"

:: Switch to project directory
set "PROJECT_DIR=%%~dp0svc"
if not exist "%%PROJECT_DIR%%" (
    echo Error: Project directory %%PROJECT_DIR%% not found!
    pause
    exit /b 1
)
cd /d "%%PROJECT_DIR%%"

echo ==============================================
echo SVC Environment Activated!
echo Current Python version:
"%%PYTHON_EXE%%" --version
echo Current pip version:
"%%PYTHON_EXE%%" -m pip --version
echo ==============================================
echo Available commands:
echo 1. Inference: "%%PYTHON_EXE%%" inference.py
echo 2. Web Service: "%%PYTHON_EXE%%" app.py
echo 3. ONNX Inference: "%%PYTHON_EXE%%" inference_onnx.py
echo 4. Real-time GUI: "%%PYTHON_EXE%%" real-time-gui.py
echo ==============================================
:: Keep window open
cmd /k
""".encode('utf-8')
        f.write(content)

    print("OK: Startup script created")

    # Create README documentation
    print("Creating README documentation...")
    readme_path = os.path.join(TARGET_DIR, "README_PACKAGE.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""# SVC Package Instructions

## Included Content
- Complete Python 3.10 environment (includes torch 2.4.1, onnx 1.20.0 and all dependencies)
- SVC project code and models
- One-click startup script

## Usage
1. Copy the entire directory to the target Windows PC
2. Double-click `run_svc.bat` to start the environment
3. Execute the corresponding commands in the opened command line

## Supported Commands
- `python inference.py` - Basic Inference
- `python app.py` - Start Web Service
- `python inference_onnx.py` - ONNX Inference
- `python real-time-gui.py` - Real-time GUI

## Target Platform
- Windows 10/11 (AMD/NVIDIA graphics card)
- No additional dependencies required
- CUDA automatically detected (if NVIDIA graphics card is available)
""")
    print("OK: README documentation created")

    print("==============================================")
    print("Packaging completed!")
    print("Package directory:", TARGET_DIR)
    print("==============================================")
    input("Press any key to exit...")

except Exception as e:
    print(f"ERROR: Packaging failed - {str(e)}")
    print("Please check path and permissions are correct")
    input("Press any key to exit...")
    sys.exit(1)
