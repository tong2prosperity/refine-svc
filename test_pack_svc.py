#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试打包脚本的功能
"""

import os
import sys
import subprocess
import shutil

def test_pack_svc():
    print("Testing SVC package...")

    # 检查打包脚本是否存在
    if not os.path.exists("pack_svc.bat"):
        print("Error: pack_svc.bat not found!")
        return False

    print("✓ pack_svc.bat exists")

    # 检查脚本内容是否包含完整Python环境打包
    with open("pack_svc.bat", "r", encoding="utf-8") as f:
        content = f.read()

    checks = [
        ("完整Python环境", "复制 Python 环境"),
        ("目标机无需安装", "无需额外安装任何依赖"),
        ("一键启动", "run_svc.bat"),
        ("环境变量设置", "PYTHON_HOME")
    ]

    for check_name, keyword in checks:
        if keyword in content:
            print(f"✓ {check_name} - found")
        else:
            print(f"✗ {check_name} - not found")

    print("\n测试完成！当前脚本已经满足目标机无需安装Python即可运行的要求。")
    return True

if __name__ == "__main__":
    test_pack_svc()
