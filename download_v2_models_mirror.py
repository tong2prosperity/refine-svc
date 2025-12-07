#!/usr/bin/env python3
"""
下载 V2 模型所需的依赖模型（使用镜像站）
包括: Whisper, Hubert, BigVGAN
"""

import os
from huggingface_hub import snapshot_download
import shutil

# 设置镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_model(repo_id, local_dir, description):
    """下载模型到本地目录"""
    print(f"\n{'='*60}")
    print(f"📥 下载 {description}")
    print(f"   仓库: {repo_id}")
    print(f"   目标: {local_dir}")
    print(f"   镜像: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
    print(f"{'='*60}")
    
    try:
        # 如果目录已存在，询问是否覆盖
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"⚠️  目录已存在且不为空: {local_dir}")
            response = input("是否覆盖? (y/n): ").strip().lower()
            if response != 'y':
                print("⏭️  跳过下载")
                return True
            shutil.rmtree(local_dir)
        
        # 创建父目录
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        
        # 下载模型
        print("⏳ 正在下载... (这可能需要几分钟)")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            cache_dir="./temp_cache",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✅ 下载完成: {local_dir}")
        
        # 显示下载的文件
        files = os.listdir(local_dir)
        print(f"📁 下载的文件 ({len(files)} 个):")
        for f in files[:5]:  # 只显示前5个
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... 还有 {len(files) - 5} 个文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🚀 V2 模型依赖下载工具（镜像版）")
    print("="*60)
    print(f"\n使用镜像站点: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
    print("\n此脚本将下载以下模型:")
    print("1. Whisper-small (语音特征提取)")
    print("2. Hubert-large-ll60k (语音特征提取)")
    print("3. BigVGAN v2 22kHz (声码器)")
    print("4. ASTRAL Quantization (语音量化)")
    print("5. Seed-VC V2 Checkpoints (主模型)")
    print("6. CAMPPlus (说话人编码器)")
    print("\n总下载量: 约 4-5 GB")
    
    response = input("\n是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ 取消下载")
        return
    
    # 下载模型列表
    models = [
        {
            "repo_id": "openai/whisper-small",
            "local_dir": "./models/whisper-small",
            "description": "Whisper-small 模型"
        },
        {
            "repo_id": "facebook/hubert-large-ll60k",
            "local_dir": "./models/hubert",
            "description": "Hubert-large-ll60k 模型"
        },
        {
            "repo_id": "nvidia/bigvgan_v2_22khz_80band_256x",
            "local_dir": "./models/bigvgan",
            "description": "BigVGAN v2 22kHz 声码器"
        },
        {
            "repo_id": "Plachta/ASTRAL-quantization",
            "local_dir": "./models/astral-quantization",
            "description": "ASTRAL Quantization 模型"
        },
        {
            "repo_id": "Plachta/Seed-VC",
            "local_dir": "./models/seed-vc",
            "description": "Seed-VC V2 Checkpoints"
        },
        {
            "repo_id": "funasr/campplus",
            "local_dir": "./models/campplus",
            "description": "CAMPPlus 说话人编码器"
        }
    ]
    
    success_count = 0
    total_count = len(models)
    
    for model in models:
        if download_model(
            repo_id=model["repo_id"],
            local_dir=model["local_dir"],
            description=model["description"]
        ):
            success_count += 1
    
    # 清理临时缓存
    print(f"\n{'='*60}")
    print("🧹 清理临时文件...")
    try:
        if os.path.exists("./temp_cache"):
            shutil.rmtree("./temp_cache")
        print("✅ 临时文件清理完成")
    except Exception as e:
        print(f"⚠️  清理临时文件失败: {e}")
    
    # 显示结果
    print(f"\n{'='*60}")
    print(f"📊 下载结果")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✨ 所有模型下载完成!")
        print("\n下一步:")
        print("  运行测试命令:")
        print("  /opt/miniconda3/envs/svc/bin/python inference_v2.py \\")
        print("    --source examples/source/yae_0.wav \\")
        print("    --target examples/reference/dingzhen_0.wav \\")
        print("    --output ./output")
    else:
        print("\n⚠️  部分模型下载失败")
        print("可以尝试：")
        print("1. 单独下载失败的模型")
        print("2. 使用方案 B：直接使用在线模型")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
