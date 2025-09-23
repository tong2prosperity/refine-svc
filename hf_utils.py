from local_model_loader import load_local_model


hf_cache_dir = "./models"

repo_map = {
    "Plachta/Seed-VC": {
        "model_filename": hf_cache_dir + "/seed-vc/DiT_uvit_tat_xlsr_ema.pth",
        "config_filename": hf_cache_dir + "/seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml"
    },
    "funasr/campplus": {
        "model_filename": hf_cache_dir + "/funasr-campplus/campplus_cn_common.bin",
        "config_filename": None
    },
    "FunAudioLLM/CosyVoice-300M": {
        "model_filename": hf_cache_dir + "/funaudio-cosyvoice/hift.pt",
        "config_filename": None
    }   
}

def load_custom_model_from_hf_map(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    if repo_id in repo_map:
        if repo_map[repo_id]["config_filename"] is None:
            return repo_map[repo_id]["model_filename"]
        return repo_map[repo_id]["model_filename"], repo_map[repo_id]["config_filename"]
    else:
        return load_custom_model_from_hf(repo_id, model_filename, config_filename)

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    """
    Load model from local directory instead of HuggingFace Hub.
    This function maintains the same interface as the original HF version.
    """
    return load_local_model(repo_id, model_filename, config_filename)