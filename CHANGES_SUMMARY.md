# 项目修改总结 - 取消 HuggingFace 下载依赖

## 修改概述

本项目已成功修改为使用本地模型加载，完全取消了对外部 HuggingFace Hub 的依赖。所有模型现在都从本地目录加载。

## 主要修改内容

### 1. 新增文件

- **`local_model_loader.py`**: 本地模型加载器，替换了 HuggingFace Hub 下载功能
- **`download_models.py`**: 模型下载脚本，帮助用户下载所需模型到本地
- **`setup_local_models.py`**: 设置向导，指导用户完成本地模型配置
- **`MODEL_SETUP.md`**: 详细的模型设置指南
- **`CHANGES_SUMMARY.md`**: 本文件，记录所有修改

### 2. 修改的文件

#### 核心文件
- **`hf_utils.py`**: 修改为使用本地模型加载器
- **`requirements.txt`**: 移除了 `huggingface-hub` 依赖

#### 应用文件
- **`real-time-gui.py`**: 修改所有模型加载逻辑使用本地路径
- **`app_svc.py`**: 修改 Whisper、Hubert、Wav2Vec2、BigVGAN 模型加载
- **`app_vc.py`**: 修改所有 transformers 模型加载
- **`inference.py`**: 修改所有模型加载逻辑
- **`eval.py`**: 修改评估相关的模型加载

#### 包装器文件
- **`seed_vc_wrapper.py`**: 修改 Whisper 和 BigVGAN 模型加载
- **`modules/v2/vc_wrapper.py`**: 通过 `hf_utils.py` 间接使用本地加载

## 模型目录结构

```
models/
├── seed-vc/           # Seed-VC 主模型
├── campplus/          # CAMPPlus 说话人编码器
├── astral-quantization/ # ASTRAL 量化模型
├── rmvpe/             # RMVPE F0 提取器
├── cosyvoice/         # CosyVoice HiFT 声码器
├── whisper/           # Whisper 语音特征提取器
├── wav2vec2/          # Wav2Vec2 语音特征提取器
├── hubert/            # Hubert 语音特征提取器
├── bigvgan/           # BigVGAN 声码器
├── wavlm/             # WavLM 模型 (评估用)
└── funasr/            # FunASR VAD 模型缓存
```

## 运行 real-time-gui.py 所需的最小模型

### 必需模型
1. **主模型**: 
   - `models/seed-vc/DiT_uvit_tat_xlsr_ema.pth`
   - `models/seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml`

2. **CAMPPlus 说话人编码器**:
   - `models/campplus/campplus_cn_common.bin`

3. **声码器** (根据配置选择):
   - BigVGAN: `models/bigvgan/v2_22khz_80band_256x/`
   - HiFiGAN: `models/cosyvoice/hift.pt`

4. **语音特征提取器** (根据配置选择):
   - Whisper: `models/whisper/small/` (默认)
   - Wav2Vec2: `models/wav2vec2/xlsr-53/`
   - Hubert: `models/hubert/base-ls960/`

5. **VAD 模型**:
   - 通过 FunASR 自动下载到 `models/funasr/`

## 使用方法

### 快速开始
```bash
# 1. 运行设置向导
python setup_local_models.py

# 2. 下载模型
python download_models.py --all

# 3. 运行应用
python real-time-gui.py
```

### 手动设置
1. 参考 `MODEL_SETUP.md` 了解详细步骤
2. 手动下载模型文件到对应目录
3. 运行应用测试

## 兼容性

- **向后兼容**: 保持了原有的 API 接口，现有代码无需修改
- **配置兼容**: 支持所有原有的模型配置选项
- **功能完整**: 所有原有功能都得到保留

## 优势

1. **离线运行**: 完全不需要网络连接即可运行
2. **速度提升**: 避免了每次启动时的模型下载
3. **稳定性**: 不依赖外部服务的可用性
4. **可控性**: 用户可以控制使用哪个版本的模型
5. **存储优化**: 模型文件可以重复使用

## 注意事项

1. **存储空间**: 需要 10-20GB 存储空间用于模型文件
2. **首次设置**: 需要一次性下载所有模型文件
3. **版本管理**: 需要手动更新模型版本
4. **权限**: 确保对 `models/` 目录有读写权限

## 故障排除

### 常见问题
1. **模型文件未找到**: 检查文件路径和文件名是否正确
2. **权限错误**: 确保对模型目录有读写权限
3. **内存不足**: 某些大模型可能需要更多内存

### 调试方法
```bash
# 检查模型目录结构
python local_model_loader.py

# 测试模型加载
python -c "from hf_utils import load_custom_model_from_hf; print('OK')"

# 查看所需模型
python download_models.py --list
```

## 技术细节

### 模型映射
- 使用 `MODEL_MAPPING` 字典映射 HuggingFace 仓库到本地路径
- 支持单文件模型和目录模型两种类型
- 自动处理模型文件名和配置文件的映射

### 错误处理
- 提供清晰的错误信息指导用户
- 支持部分模型缺失的情况
- 自动创建必要的目录结构

### 性能优化
- 避免重复下载已存在的模型
- 支持增量下载特定模型
- 提供模型验证功能

## 后续维护

1. **模型更新**: 定期检查并更新模型版本
2. **新模型支持**: 在 `MODEL_MAPPING` 中添加新模型
3. **文档更新**: 保持文档与代码同步
4. **用户反馈**: 收集用户使用反馈并改进

---

**修改完成时间**: 2024年12月
**修改范围**: 全项目
**影响程度**: 重大 - 改变了模型加载方式
**向后兼容**: 是
