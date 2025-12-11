#!/usr/bin/env python3
import os
import shutil
import sys

# Use the Python currently running this script!
CURRENT_PYTHON_EXE = sys.executable
CURRENT_PYTHON_DIR = os.path.dirname(CURRENT_PYTHON_EXE)

print("Using Python at:", CURRENT_PYTHON_EXE)

# Create package directory
PACKAGE_DIR = os.path.join(os.getcwd(), "svc_portable")
print(f"\nCreating package at: {PACKAGE_DIR}")

if os.path.exists(PACKAGE_DIR):
    shutil.rmtree(PACKAGE_DIR)

# Copy the whole Python directory
PYTHON_PACKAGE_DIR = os.path.join(PACKAGE_DIR, "python")
print(f"\nCopying Python environment from {CURRENT_PYTHON_DIR}")
shutil.copytree(CURRENT_PYTHON_DIR, PYTHON_PACKAGE_DIR,
                ignore=lambda dir_path, contents: [f for f in contents if f in ('.git', '.idea', '__pycache__', 'svc_portable')])

# Copy project files
PROJECT_PACKAGE_DIR = os.path.join(PACKAGE_DIR, "svc")
os.makedirs(PROJECT_PACKAGE_DIR)

print("\nCopying project files")

for item in os.listdir('.'):
    if item == 'svc_portable' or item == '.git' or item == '__pycache__' or item in ('pack_svc.py', 'pack_svc_fixed.bat', 'pack_portable.py'):
        continue

    src = os.path.join('.', item)
    dst = os.path.join(PROJECT_PACKAGE_DIR, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

# Create simple startup script
startup_bat = os.path.join(PACKAGE_DIR, "start_svc.bat")
with open(startup_bat, 'w') as f:
    f.write(f"""
@echo off
chcp 65001 >nul
set "PYTHON_EXE=%~dp0python\\python.exe"
set "PROJECT_DIR=%~dp0svc"
cd /d "%PROJECT_DIR%"

echo SVC Portable Environment
echo ========================
echo Python: %PYTHON_EXE%
echo Project: %PROJECT_DIR%
echo ========================
echo.

if exist "%PYTHON_EXE%" (
    echo Environment ready! Type any SVC command below.
    echo Example: python inference.py
    echo.
    cmd /k
) else (
    echo ERROR: Python not found!
    echo Check path: %PYTHON_EXE%
    pause
)
""")

print("\n✅ Portable package created!")
print(f"Location: {PACKAGE_DIR}")
print("\nTo use:")
print("1. Copy the svc_portable folder to any Windows PC")
print("2. Double-click start_svc.bat")
print("3. Run your SVC commands")
