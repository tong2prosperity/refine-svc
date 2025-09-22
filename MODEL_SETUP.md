# 本地模型设置指南

本项目已修改为使用本地模型加载，不再依赖 HuggingFace Hub 下载。请按照以下步骤设置本地模型。

## 目录结构

```
models/
├── seed-vc/                    # Seed-VC 主模型
│   ├── DiT_uvit_tat_xlsr_ema.pth
│   ├── config_dit_mel_seed_uvit_xlsr_tiny.yml
│   ├── DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
│   ├── config_dit_mel_seed_uvit_whisper_small_wavenet.yml
│   ├── DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
│   ├── config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
│   ├── DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth
│   └── v2/
│       ├── cfm_small.pth
│       └── ar_base.pth
├── campplus/                   # CAMPPlus 说话人编码器
│   └── campplus_cn_common.bin
├── astral-quantization/        # ASTRAL 量化模型
│   ├── bsq32/
│   │   └── bsq32_light.pth
│   └── bsq2048/
│       └── bsq2048_light.pth
├── rmvpe/                      # RMVPE F0 提取器
│   └── rmvpe.pt
├── cosyvoice/                  # CosyVoice HiFT 声码器
│   └── hift.pt
├── whisper/                    # Whisper 语音特征提取器
│   ├── tiny/
│   ├── base/
│   ├── small/
│   ├── medium/
│   ├── large/
│   ├── large-v2/
│   └── large-v3/
├── wav2vec2/                   # Wav2Vec2 语音特征提取器
│   ├── base/
│   ├── large/
│   ├── xlsr-53/
│   └── xlsr-128/
├── hubert/                     # Hubert 语音特征提取器
│   ├── base-ls960/
│   ├── large-ls960/
│   └── xlarge-ls960/
├── bigvgan/                    # BigVGAN 声码器
│   ├── v2_22khz_80band_256x/
│   └── v2_44khz_128band_512x/
├── wavlm/                      # WavLM 模型 (用于评估)
│   └── wavlm-base-plus-sv/
└── funasr/                     # FunASR VAD 模型缓存
    └── (自动生成)
```

## 快速设置

### 方法1: 使用下载脚本 (推荐)

```bash
# 下载所有必需模型
python download_models.py --all

# 或者下载特定模型
python download_models.py --models Plachta/Seed-VC funasr/campplus

# 查看需要下载的模型列表
python download_models.py --list
```

### 方法2: 手动下载

1. **Seed-VC 模型**:
   ```bash
   # 创建目录
   mkdir -p models/seed-vc
   
   # 下载模型文件 (需要手动从 HuggingFace 下载)
   # 访问 https://huggingface.co/Plachta/Seed-VC
   # 下载所需文件到 models/seed-vc/ 目录
   ```

2. **其他模型**:
   - 访问相应的 HuggingFace 仓库
   - 下载模型文件到对应的本地目录

## 运行 real-time-gui.py 所需的最小模型

对于 `real-time-gui.py`，你需要以下模型：

### 必需模型
1. **主模型**: `models/seed-vc/DiT_uvit_tat_xlsr_ema.pth` 和 `config_dit_mel_seed_uvit_xlsr_tiny.yml`
2. **CAMPPlus**: `models/campplus/campplus_cn_common.bin`
3. **声码器**: 根据配置选择 BigVGAN 或 HiFiGAN
4. **语音特征提取器**: 根据配置选择 Whisper、Wav2Vec2 或 Hubert
5. **VAD 模型**: 通过 FunASR 自动下载到 `models/funasr/`

### 可选模型
- 其他 Seed-VC 变体模型
- 不同尺寸的 Whisper 模型
- 其他声码器模型

## 验证设置

运行以下命令验证模型是否正确设置：

```bash
python local_model_loader.py
```

这将创建目录结构并显示所需的模型文件。

## 故障排除

### 模型文件未找到
如果遇到 "Model file not found" 错误：
1. 检查文件路径是否正确
2. 确认模型文件已下载到正确位置
3. 检查文件名是否与映射表中的名称匹配

### 目录结构问题
如果目录结构不正确：
1. 运行 `python local_model_loader.py` 创建标准目录结构
2. 使用 `python download_models.py --list` 查看正确的文件映射

### 权限问题
确保对 `models/` 目录有读写权限：
```bash
chmod -R 755 models/
```

## 注意事项

1. **存储空间**: 所有模型文件大约需要 10-20GB 存储空间
2. **网络**: 首次下载需要稳定的网络连接
3. **版本兼容**: 确保下载的模型版本与代码兼容
4. **缓存**: FunASR 模型会自动缓存到 `models/funasr/` 目录

## 更新模型

要更新模型到新版本：
1. 备份当前模型文件
2. 下载新版本模型
3. 替换对应文件
4. 测试确保兼容性
