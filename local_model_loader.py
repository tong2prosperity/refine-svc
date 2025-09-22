import os

# Define local model directory structure
LOCAL_MODELS_DIR = "./models"
MODEL_MAPPING = {
    # Seed-VC models
    "Plachta/Seed-VC": {
        "DiT_uvit_tat_xlsr_ema.pth": "seed-vc/DiT_uvit_tat_xlsr_ema.pth",
        "config_dit_mel_seed_uvit_xlsr_tiny.yml": "seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth": "seed-vc/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml": "seed-vc/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth": "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml": "seed-vc/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth": "seed-vc/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
        "v2/cfm_small.pth": "seed-vc/v2/cfm_small.pth",
        "v2/ar_base.pth": "seed-vc/v2/ar_base.pth",
    },
    # CAMPPlus models
    "funasr/campplus": {
        "campplus_cn_common.bin": "campplus/campplus_cn_common.bin",
    },
    # ASTRAL quantization models
    "Plachta/ASTRAL-quantization": {
        "bsq32/bsq32_light.pth": "astral-quantization/bsq32/bsq32_light.pth",
        "bsq2048/bsq2048_light.pth": "astral-quantization/bsq2048/bsq2048_light.pth",
    },
    # RMVPE models
    "lj1995/VoiceConversionWebUI": {
        "rmvpe.pt": "rmvpe/rmvpe.pt",
    },
    # CosyVoice models
    "FunAudioLLM/CosyVoice-300M": {
        "hift.pt": "cosyvoice/hift.pt",
    },
    # Whisper models
    "openai/whisper-tiny": "whisper/tiny",
    "openai/whisper-base": "whisper/base",
    "openai/whisper-small": "whisper/small",
    "openai/whisper-medium": "whisper/medium",
    "openai/whisper-large": "whisper/large",
    "openai/whisper-large-v2": "whisper/large-v2",
    "openai/whisper-large-v3": "whisper/large-v3",
    # Wav2Vec2 models
    "facebook/wav2vec2-base": "wav2vec2/base",
    "facebook/wav2vec2-large": "wav2vec2/large",
    "facebook/wav2vec2-xlsr-53": "wav2vec2/xlsr-53",
    "facebook/wav2vec2-xlsr-128": "wav2vec2/xlsr-128",
    # Hubert models
    "facebook/hubert-base-ls960": "hubert/base-ls960",
    "facebook/hubert-large-ls960": "hubert/large-ls960",
    "facebook/hubert-xlarge-ls960": "hubert/xlarge-ls960",
    # BigVGAN models
    "nvidia/bigvgan_v2_22khz_80band_256x": "bigvgan/v2_22khz_80band_256x",
    "nvidia/bigvgan_v2_44khz_128band_512x": "bigvgan/v2_44khz_128band_512x",
}

def load_local_model(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    """
    Load model from local directory instead of HuggingFace Hub.
    
    Args:
        repo_id: The repository ID (used for mapping to local path)
        model_filename: The model filename
        config_filename: The config filename (optional)
    
    Returns:
        model_path: Path to the model file
        config_path: Path to the config file (if config_filename provided)
    """
    # Ensure models directory exists
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    # Get the local path mapping
    if repo_id in MODEL_MAPPING:
        if isinstance(MODEL_MAPPING[repo_id], dict):
            # Repository with multiple files
            if model_filename in MODEL_MAPPING[repo_id]:
                model_path = os.path.join(LOCAL_MODELS_DIR, MODEL_MAPPING[repo_id][model_filename])
            else:
                raise FileNotFoundError(f"Model file '{model_filename}' not found in mapping for '{repo_id}'")
        else:
            # Repository with single directory
            model_path = os.path.join(LOCAL_MODELS_DIR, MODEL_MAPPING[repo_id], model_filename)
    else:
        # Fallback: assume repo_id is a direct path
        model_path = os.path.join(LOCAL_MODELS_DIR, repo_id, model_filename)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if config_filename is None:
        return model_path
    
    # Handle config file
    if repo_id in MODEL_MAPPING:
        if isinstance(MODEL_MAPPING[repo_id], dict):
            if config_filename in MODEL_MAPPING[repo_id]:
                config_path = os.path.join(LOCAL_MODELS_DIR, MODEL_MAPPING[repo_id][config_filename])
            else:
                raise FileNotFoundError(f"Config file '{config_filename}' not found in mapping for '{repo_id}'")
        else:
            config_path = os.path.join(LOCAL_MODELS_DIR, MODEL_MAPPING[repo_id], config_filename)
    else:
        config_path = os.path.join(LOCAL_MODELS_DIR, repo_id, config_filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return model_path, config_path

def get_model_directory_structure():
    """
    Get the expected directory structure for local models.
    
    Returns:
        dict: Directory structure with descriptions
    """
    structure = {
        "models/": "Root directory for all local models",
        "models/seed-vc/": "Seed-VC model files and configs",
        "models/campplus/": "CAMPPlus speaker encoder models",
        "models/astral-quantization/": "ASTRAL quantization models",
        "models/rmvpe/": "RMVPE F0 extraction models",
        "models/cosyvoice/": "CosyVoice HiFT vocoder models",
        "models/whisper/": "Whisper models for speech feature extraction",
        "models/wav2vec2/": "Wav2Vec2 models for speech feature extraction",
        "models/hubert/": "Hubert models for speech feature extraction",
        "models/bigvgan/": "BigVGAN vocoder models",
    }
    return structure

def create_model_directories():
    """Create the model directory structure."""
    structure = get_model_directory_structure()
    for directory, description in structure.items():
        os.makedirs(os.path.join(LOCAL_MODELS_DIR, directory), exist_ok=True)
        print(f"Created directory: {directory} - {description}")

def print_model_requirements():
    """Print the required model files and their sources."""
    print("Required model files for local loading:")
    print("=" * 50)
    
    for repo_id, files in MODEL_MAPPING.items():
        if isinstance(files, dict):
            print(f"\n{repo_id}:")
            for filename, local_path in files.items():
                print(f"  - {filename} -> {local_path}")
        else:
            print(f"\n{repo_id}:")
            print(f"  - Directory: {files}")
    
    print("\n" + "=" * 50)
    print("To download these models, you can:")
    print("1. Use huggingface-hub to download them once")
    print("2. Manually download from HuggingFace and place in the correct directories")
    print("3. Use the provided download script (if available)")

if __name__ == "__main__":
    print_model_requirements()
    create_model_directories()
