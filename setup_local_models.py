#!/usr/bin/env python3
"""
Setup script for local model loading.
This script helps users set up the local model directory structure and provides guidance.
"""

import os
import sys
from pathlib import Path
from local_model_loader import create_model_directories, print_model_requirements, MODEL_MAPPING

def main():
    print("=" * 60)
    print("Seed-VC 本地模型设置向导")
    print("=" * 60)
    
    # Create model directories
    print("\n1. 创建模型目录结构...")
    create_model_directories()
    print("✓ 目录结构创建完成")
    
    # Show required models
    print("\n2. 显示所需模型列表...")
    print_model_requirements()
    
    # Check for existing models
    print("\n3. 检查现有模型...")
    existing_models = []
    missing_models = []
    
    for repo_id, files in MODEL_MAPPING.items():
        if isinstance(files, dict):
            for filename, local_path in files.items():
                full_path = os.path.join("./models", local_path)
                if os.path.exists(full_path):
                    existing_models.append(f"{repo_id}/{filename}")
                    print(f"✓ {filename}")
                else:
                    missing_models.append(f"{repo_id}/{filename}")
                    print(f"✗ {filename}")
        else:
            # Directory-based models
            local_dir = os.path.join("./models", files)
            if os.path.exists(local_dir) and os.listdir(local_dir):
                existing_models.append(repo_id)
                print(f"✓ {repo_id} (目录)")
            else:
                missing_models.append(repo_id)
                print(f"✗ {repo_id} (目录)")
    
    print(f"\n现有模型: {len(existing_models)}")
    print(f"缺失模型: {len(missing_models)}")
    
    if missing_models:
        print("\n4. 下载缺失模型...")
        print("请选择下载方式:")
        print("1. 使用下载脚本 (推荐)")
        print("2. 手动下载")
        print("3. 跳过下载")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            print("\n运行下载脚本...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, "download_models.py", "--all"], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ 模型下载完成")
                else:
                    print(f"✗ 下载失败: {result.stderr}")
            except Exception as e:
                print(f"✗ 运行下载脚本失败: {e}")
        elif choice == "2":
            print("\n手动下载指南:")
            print("1. 访问 HuggingFace 仓库")
            print("2. 下载模型文件到对应目录")
            print("3. 参考 MODEL_SETUP.md 了解详细步骤")
        else:
            print("跳过下载，请稍后手动设置模型")
    
    # Test model loading
    print("\n5. 测试模型加载...")
    try:
        from hf_utils import load_custom_model_from_hf
        # Test loading a simple model
        if os.path.exists("./models/campplus/campplus_cn_common.bin"):
            test_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin")
            print(f"✓ 模型加载测试成功: {test_path}")
        else:
            print("✗ 无法测试模型加载 - 缺少测试模型")
    except Exception as e:
        print(f"✗ 模型加载测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("设置完成!")
    print("=" * 60)
    print("\n下一步:")
    print("1. 确保所有必需模型已下载")
    print("2. 运行 python real-time-gui.py 测试")
    print("3. 如有问题，请参考 MODEL_SETUP.md")
    
    # Show real-time-gui specific requirements
    print("\nreal-time-gui.py 所需的最小模型:")
    print("- models/seed-vc/DiT_uvit_tat_xlsr_ema.pth")
    print("- models/seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml")
    print("- models/campplus/campplus_cn_common.bin")
    print("- 声码器模型 (根据配置)")
    print("- 语音特征提取器模型 (根据配置)")

if __name__ == "__main__":
    main()
