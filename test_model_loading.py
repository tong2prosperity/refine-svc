#!/usr/bin/env python3
"""
快速测试 V2 模型加载
用于验证模型检查点能否正常加载和初始化
"""
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig
import argparse

# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

def test_model_loading(ar_checkpoint_path=None, cfm_checkpoint_path=None):
    """测试模型加载"""
    print("="*50)
    print("🔍 开始测试 V2 模型加载")
    print("="*50)
    
    print(f"\n📱 使用设备: {device}")
    print(f"🔢 数据类型: {dtype}")
    
    try:
        # 1. 加载配置
        print("\n📄 步骤 1/4: 加载配置文件...")
        cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
        print("✅ 配置文件加载成功")
        
        # 2. 实例化模型包装器
        print("\n🏗️  步骤 2/4: 实例化模型包装器...")
        vc_wrapper = instantiate(cfg)
        print("✅ 模型包装器实例化成功")
        
        # 3. 加载检查点
        print("\n💾 步骤 3/4: 加载模型检查点...")
        if ar_checkpoint_path:
            print(f"   - AR 检查点: {ar_checkpoint_path}")
        else:
            print(f"   - AR 检查点: 使用默认 (从 HuggingFace 下载)")
        
        if cfm_checkpoint_path:
            print(f"   - CFM 检查点: {cfm_checkpoint_path}")
        else:
            print(f"   - CFM 检查点: 使用默认 (从 HuggingFace 下载)")
        
        vc_wrapper.load_checkpoints(
            ar_checkpoint_path=ar_checkpoint_path,
            cfm_checkpoint_path=cfm_checkpoint_path
        )
        print("✅ 模型检查点加载成功")
        
        # 4. 转移到设备并设置为评估模式
        print(f"\n🚀 步骤 4/4: 将模型转移到 {device} 并设置为评估模式...")
        vc_wrapper.to(device)
        vc_wrapper.eval()
        print("✅ 模型准备完成")
        
        # 5. 设置 AR 缓存
        print("\n🔧 设置 AR 模型缓存...")
        vc_wrapper.setup_ar_caches(
            max_batch_size=1,
            max_seq_len=4096,
            dtype=dtype,
            device=device
        )
        print("✅ AR 缓存设置完成")
        
        # 6. 显示模型信息
        print("\n" + "="*50)
        print("📊 模型信息")
        print("="*50)
        
        # 统计参数数量
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        if hasattr(vc_wrapper, 'ar'):
            ar_params = count_parameters(vc_wrapper.ar)
            print(f"AR 模型参数: {ar_params:,} ({ar_params/1e6:.1f}M)")
        
        if hasattr(vc_wrapper, 'cfm'):
            cfm_params = count_parameters(vc_wrapper.cfm)
            print(f"CFM 模型参数: {cfm_params:,} ({cfm_params/1e6:.1f}M)")
        
        if hasattr(vc_wrapper, 'vocoder'):
            vocoder_params = count_parameters(vc_wrapper.vocoder)
            print(f"Vocoder 参数: {vocoder_params:,} ({vocoder_params/1e6:.1f}M)")
        
        total_params = count_parameters(vc_wrapper)
        print(f"总参数: {total_params:,} ({total_params/1e6:.1f}M)")
        
        print("\n" + "="*50)
        print("✨ 模型加载测试成功！")
        print("="*50)
        print("\n💡 提示: 模型已成功加载，可以进行推理测试")
        print("   - 使用 Web UI: python app_vc_v2.py")
        print("   - 使用命令行: python inference_v2.py --source <源音频> --target <参考音频>")
        
        return True
        
    except Exception as e:
        print("\n" + "="*50)
        print("❌ 模型加载失败！")
        print("="*50)
        print(f"\n错误信息: {str(e)}")
        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 V2 模型加载")
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="AR 模型检查点路径 (可选)")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="CFM 模型检查点路径 (可选)")
    args = parser.parse_args()
    
    test_model_loading(args.ar_checkpoint_path, args.cfm_checkpoint_path)
