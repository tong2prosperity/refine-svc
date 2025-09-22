from local_model_loader import load_local_model


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    """
    Load model from local directory instead of HuggingFace Hub.
    This function maintains the same interface as the original HF version.
    """
    return load_local_model(repo_id, model_filename, config_filename)