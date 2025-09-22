#!/usr/bin/env python3
"""
Script to download all required models for local use.
This script downloads models from HuggingFace Hub and organizes them in the local directory structure.
"""

import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
from local_model_loader import MODEL_MAPPING, LOCAL_MODELS_DIR

def download_single_file(repo_id, filename, local_path):
    """Download a single file from HuggingFace Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="./temp_cache"
        )
        
        # Create target directory
        target_dir = os.path.dirname(local_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file to target location
        import shutil
        shutil.copy2(downloaded_path, local_path)
        print(f"✓ Downloaded to {local_path}")
        return True
    except (FileNotFoundError, ConnectionError, RuntimeError) as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def download_whisper_model(model_name, local_dir):
    """Download Whisper model and its files."""
    print(f"Downloading Whisper model {model_name}...")
    try:
        # Download the entire model directory
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir="./temp_cache"
        )
        
        # Copy to local directory
        import shutil
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        shutil.copytree(snapshot_path, local_dir)
        print(f"✓ Downloaded Whisper model to {local_dir}")
        return True
    except (FileNotFoundError, ConnectionError, RuntimeError) as e:
        print(f"✗ Failed to download Whisper model {model_name}: {e}")
        return False

def download_bigvgan_model(model_name, local_dir):
    """Download BigVGAN model."""
    print(f"Downloading BigVGAN model {model_name}...")
    try:
        # Download the entire model directory
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir="./temp_cache"
        )
        
        # Copy to local directory
        import shutil
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        shutil.copytree(snapshot_path, local_dir)
        print(f"✓ Downloaded BigVGAN model to {local_dir}")
        return True
    except (FileNotFoundError, ConnectionError, RuntimeError) as e:
        print(f"✗ Failed to download BigVGAN model {model_name}: {e}")
        return False

def download_models(selected_models=None):
    """Download all required models."""
    print("Starting model download process...")
    print("=" * 60)
    
    # Create models directory
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # Download models from MODEL_MAPPING
    for repo_id, files in MODEL_MAPPING.items():
        if isinstance(files, dict):
            # Repository with multiple files
            for filename, local_path in files.items():
                if selected_models is None or repo_id in selected_models:
                    total_count += 1
                    full_local_path = os.path.join(LOCAL_MODELS_DIR, local_path)
                    if download_single_file(repo_id, filename, full_local_path):
                        success_count += 1
        else:
            # Single directory (for Whisper, BigVGAN, etc.)
            if selected_models is None or repo_id in selected_models:
                total_count += 1
                local_dir = os.path.join(LOCAL_MODELS_DIR, files)
                
                if "whisper" in repo_id:
                    if download_whisper_model(repo_id, local_dir):
                        success_count += 1
                elif "bigvgan" in repo_id:
                    if download_bigvgan_model(repo_id, local_dir):
                        success_count += 1
                else:
                    # For other models, try to download the entire repository
                    try:
                        snapshot_path = snapshot_download(
                            repo_id=repo_id,
                            cache_dir="./temp_cache"
                        )
                        import shutil
                        if os.path.exists(local_dir):
                            shutil.rmtree(local_dir)
                        shutil.copytree(snapshot_path, local_dir)
                        print(f"✓ Downloaded {repo_id} to {local_dir}")
                        success_count += 1
                    except (FileNotFoundError, ConnectionError, RuntimeError) as e:
                        print(f"✗ Failed to download {repo_id}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Download completed: {success_count}/{total_count} models downloaded successfully")
    
    # Clean up temp cache
    try:
        import shutil
        if os.path.exists("./temp_cache"):
            shutil.rmtree("./temp_cache")
        print("Cleaned up temporary cache")
    except OSError:
        pass

def list_required_models():
    """List all required models."""
    print("Required models for local loading:")
    print("=" * 50)
    
    for repo_id, files in MODEL_MAPPING.items():
        if isinstance(files, dict):
            print(f"\n{repo_id}:")
            for filename, local_path in files.items():
                print(f"  - {filename}")
        else:
            print(f"\n{repo_id}:")
            print(f"  - Directory: {files}")

def main():
    parser = argparse.ArgumentParser(description="Download models for local use")
    parser.add_argument("--list", action="store_true", help="List all required models")
    parser.add_argument("--models", nargs="+", help="Download specific models only")
    parser.add_argument("--all", action="store_true", help="Download all models")
    
    args = parser.parse_args()
    
    if args.list:
        list_required_models()
    elif args.all or args.models:
        selected_models = args.models if args.models else None
        download_models(selected_models)
    else:
        print("Use --list to see required models, --all to download all, or --models <repo_id> to download specific models")
        print("Example: python download_models.py --models Plachta/Seed-VC funasr/campplus")

if __name__ == "__main__":
    main()
