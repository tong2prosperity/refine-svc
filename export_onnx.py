import argparse
import os
import sys
import torch
# Patch for transformers import error with some torch versions
if not hasattr(torch, 'float8_e4m3fn'):
    try:
        class Float8_e4m3fn: pass
        torch.float8_e4m3fn = Float8_e4m3fn
    except:
        pass

# Patch transformers.modeling_utils for accelerate compatibility
try:
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.utils import find_tied_parameters
    import transformers.modeling_utils
    if not hasattr(transformers.modeling_utils, 'init_empty_weights'):
        transformers.modeling_utils.init_empty_weights = init_empty_weights
    if not hasattr(transformers.modeling_utils, 'find_tied_parameters'):
        transformers.modeling_utils.find_tied_parameters = find_tied_parameters
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to patch transformers: {e}")

import yaml
import json
from pathlib import Path
import logging

# Add current directory to sys.path
sys.path.append(os.getcwd())

from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return recursive_munch(config)

def export_campplus(model, output_dir, device="cpu"):
    logger.info("Exporting CAMPPlus...")
    model.eval().to(device)
    
    # Dummy input: (Batch, Length, MelBins) -> (1, 200, 80)
    dummy_input = torch.randn(1, 200, 80).to(device)
    dummy_lens = torch.tensor([200]).to(device)
    
    output_path = os.path.join(output_dir, "campplus.onnx")
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_lens),
        output_path,
        input_names=["speech", "speech_lengths"],
        output_names=["embedding"],
        dynamic_axes={
            "speech": {0: "batch_size", 1: "time"},
            "speech_lengths": {0: "batch_size"},
        },
        opset_version=14
    )
    logger.info(f"CAMPPlus exported to {output_path}")

def export_dit(model, output_dir, device="cpu"):
    logger.info("Exporting DiT...")
    model.eval().to(device)
    
    B = 1
    C = model.in_channels
    T = 100
    style_dim = 192
    
    dummy_x = torch.randn(B, C, T).to(device)
    dummy_prompt_x = torch.randn(B, C, T).to(device)
    dummy_x_lens = torch.tensor([T], dtype=torch.long).to(device)
    dummy_t = torch.tensor([0.5]).to(device)
    dummy_style = torch.randn(B, style_dim).to(device)
    
    content_dim = model.content_dim
    dummy_cond = torch.randn(B, T, content_dim).to(device)
    
    output_path = os.path.join(output_dir, "dit.onnx")
    
    torch.onnx.export(
        model,
        (dummy_x, dummy_prompt_x, dummy_x_lens, dummy_t, dummy_style, dummy_cond),
        output_path,
        input_names=["x", "prompt_x", "x_lens", "t", "style", "cond"],
        output_names=["dphi_dt"],
        dynamic_axes={
            "x": {0: "batch_size", 2: "time"},
            "prompt_x": {0: "batch_size", 2: "time"},
            "x_lens": {0: "batch_size"},
            "t": {0: "batch_size"},
            "style": {0: "batch_size"},
            "cond": {0: "batch_size", 1: "time"},
        },
        opset_version=14
    )
    logger.info(f"DiT exported to {output_path}")

def export_length_regulator(model, output_dir, device="cpu"):
    logger.info("Exporting LengthRegulator...")
    model.eval().to(device)
    
    B = 1
    T = 100
    in_channels = model.content_in_proj.in_features if not model.is_discrete else 1
    
    dummy_x = torch.randn(B, T, in_channels).to(device)
    dummy_ylens = torch.tensor([T], dtype=torch.long).to(device)
    
    dummy_f0 = None
    if model.f0_condition:
        dummy_f0 = torch.randn(B, T).to(device)
        
    args = (dummy_x, dummy_ylens, None, dummy_f0)
    input_names = ["x", "ylens", "n_quantizers", "f0"]
    
    output_path = os.path.join(output_dir, "length_regulator.onnx")
    
    torch.onnx.export(
        model,
        args,
        output_path,
        input_names=input_names,
        output_names=["out", "olens"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "time"},
            "ylens": {0: "batch_size"},
            "f0": {0: "batch_size", 1: "time"},
        },
        opset_version=14
    )
    logger.info(f"LengthRegulator exported to {output_path}")

def export_vocoder(model, output_dir, device="cpu"):
    logger.info("Exporting Vocoder...")
    model.eval().to(device)
    
    is_hifigan = hasattr(model, 'istft_params')
    
    B = 1
    T = 100
    
    if is_hifigan:
        C = model.conv_pre.in_channels
        dummy_x = torch.randn(B, C, T).to(device)
        # dummy_f0 = torch.randn(B, T).to(device)
        args = (dummy_x,)
        input_names = ["mel"]
        output_names = ["real", "imag"] # Changed output names
        dynamic_axes = {
            "mel": {0: "batch_size", 2: "time"},
        }
        opset = 17 # Use 17 for STFT support
        
        # Monkeypatch forward and _stft
        import types
        import torch.nn.functional as F
        
        def custom_stft(self, x):
            spec = torch.stft(
                x,
                self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
                return_complex=False)
            return spec[..., 0], spec[..., 1]

        def custom_forward(self, x: torch.Tensor, f0=None) -> torch.Tensor:
            if f0 is None:
                f0 = self.f0_predictor(x)
            s = self._f02source(f0)

            s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, self.lrelu_slope)
                x = self.ups[i](x)

                if i == self.num_upsamples - 1:
                    x = self.reflection_pad(x)

                # fusion
                si = self.source_downs[i](s_stft)
                si = self.source_resblocks[i](si)
                x = x + si

                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels

            x = F.leaky_relu(x)
            x = self.conv_post(x)
            magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
            phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])
            
            real = magnitude * torch.cos(phase)
            img = magnitude * torch.sin(phase)
            return real, img

        model._stft = types.MethodType(custom_stft, model)
        model.forward = types.MethodType(custom_forward, model)
        
    else:
        C = model.conv_pre.in_channels
        dummy_x = torch.randn(B, C, T).to(device)
        args = (dummy_x,)
        input_names = ["mel"]
        output_names = ["audio"]
        dynamic_axes = {
            "mel": {0: "batch_size", 2: "time"},
        }
        opset = 14

    output_path = os.path.join(output_dir, "vocoder.onnx")
    
    torch.onnx.export(
        model,
        args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset
    )
    logger.info(f"Vocoder exported to {output_path}")

def export_speech_tokenizer(model_type, config, output_dir, device="cpu"):
    logger.info(f"Exporting Speech Tokenizer ({model_type})...")
    
    if model_type == 'whisper':
        from transformers import WhisperModel
        name = config['model_params']['speech_tokenizer']['name']
        model = WhisperModel.from_pretrained(name).to(device)
        encoder = model.encoder
        encoder.eval()
        
        dummy_input = torch.randn(1, 80, 3000).to(device)
        
        output_path = os.path.join(output_dir, "speech_tokenizer.onnx")
        
        torch.onnx.export(
            encoder,
            (dummy_input,),
            output_path,
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_features": {0: "batch_size", 2: "time"},
            },
            opset_version=14
        )
        logger.info(f"Speech Tokenizer exported to {output_path}")
        
    elif model_type == 'xlsr':
        from transformers import Wav2Vec2Model
        name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        
        local_path = "./models/wav2vec2"
        if os.path.exists(local_path):
            model_path = local_path
        else:
            model_path = name
            
        model = Wav2Vec2Model.from_pretrained(model_path).to(device)
        model.encoder.layers = model.encoder.layers[:output_layer]
        model.eval()
        
        dummy_input = torch.randn(1, 16000).to(device)
        
        output_path = os.path.join(output_dir, "speech_tokenizer.onnx")
        
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=["audio"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "audio": {0: "batch_size", 1: "time"},
            },
            opset_version=14
        )
        logger.info(f"Speech Tokenizer exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to DiT checkpoint")
    parser.add_argument("--config_path", type=str, default="", help="Path to DiT config")
    parser.add_argument("--output_dir", type=str, default="onnx_models", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load DiT and Config
    if not args.checkpoint_path:
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf_map(
            "Plachta/Seed-VC",
            "DiT_uvit_tat_xlsr_ema.pth",
            "config_dit_mel_seed_uvit_xlsr_tiny.yml"
        )
    else:
        dit_checkpoint_path = args.checkpoint_path
        dit_config_path = args.config_path
        
    logger.info(f"Loading config from {dit_config_path}")
    config = load_config(dit_config_path)
    model_params = config.model_params
    model_params.dit_type = 'DiT'
    
    # Build DiT
    nets = build_model(model_params, stage="DiT")
    dit = nets.cfm.estimator
    length_regulator = nets.length_regulator
    
    # Initialize caches
    dit.setup_caches(max_batch_size=1, max_seq_length=8192)
    
    # Load Checkpoint
    logger.info(f"Loading checkpoint from {dit_checkpoint_path}")
    load_checkpoint(nets, None, dit_checkpoint_path, load_only_params=True, ignore_modules=[], is_distributed=False)
    
    # Export DiT
    export_dit(dit, args.output_dir, device)
    
    # Export LengthRegulator
    export_length_regulator(length_regulator, args.output_dir, device)
    
    # Load and Export CAMPPlus
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf_map(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    export_campplus(campplus_model, args.output_dir, device)
    
    # Load and Export Vocoder
    vocoder_type = model_params.vocoder.type
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        export_vocoder(bigvgan_model, args.output_dir, device)
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        with open('configs/hifigan.yml', 'r') as f:
            hift_config = yaml.safe_load(f)
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf_map("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        export_vocoder(hift_gen, args.output_dir, device)
    
    # Load and Export Speech Tokenizer
    speech_tokenizer_type = model_params.speech_tokenizer.type
    export_speech_tokenizer(speech_tokenizer_type, config, args.output_dir, device)
    
    # Load and Export RMVPE
    from modules.rmvpe import RMVPE
    rmvpe_path = "models/rmvpe/rmvpe.pt" # Assuming standard path
    if os.path.exists(rmvpe_path):
        rmvpe_model = RMVPE(rmvpe_path, is_half=False, device=device)
        export_rmvpe(rmvpe_model, args.output_dir, device)
    else:
        logger.warning(f"RMVPE checkpoint not found at {rmvpe_path}, skipping export.")

def export_rmvpe(rmvpe_instance, output_dir, device="cpu"):
    logger.info("Exporting RMVPE...")
    
    class RMVPEWrapper(torch.nn.Module):
        def __init__(self, rmvpe):
            super().__init__()
            self.mel_extractor = rmvpe.mel_extractor
            self.model = rmvpe.model
            
            # Initialize custom STFT for ONNX export compatibility
            from modules.rmvpe import STFT
            self.stft = STFT(
                filter_length=self.mel_extractor.n_fft,
                hop_length=self.mel_extractor.hop_length,
                win_length=self.mel_extractor.win_length,
                window="hann"
            ).to(device)
            
        def forward(self, audio):
            # audio: (B, T)
            
            # Custom STFT logic reimplemented using Conv1d for ONNX compatibility
            # 1. Pad with explicit 3D reshape for reflect mode support
            audio_padded = audio.unsqueeze(1) # (B, 1, T)
            audio_padded = torch.nn.functional.pad(
                audio_padded,
                (self.stft.pad_amount, self.stft.pad_amount),
                mode="reflect",
            )
            # audio_padded: (B, 1, T_padded)
            
            # 2. Conv1d (equivalent to Unfold + Matmul)
            # forward_basis: (F_out, Filter_len) -> (F_out, 1, Filter_len)
            weights = self.stft.forward_basis.unsqueeze(1)
            
            forward_transform = torch.nn.functional.conv1d(
                audio_padded,
                weights,
                stride=self.stft.hop_length,
            )
            # forward_transform: (B, F_out, N_frames)
            
            # 3. Magnitude
            cutoff = int((self.stft.filter_length / 2) + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]
            magnitude = torch.sqrt(real_part**2 + imag_part**2)
            
            # Mel Basis
            mel_output = torch.matmul(self.mel_extractor.mel_basis, magnitude)
            
            # Log
            log_mel_spec = torch.log(torch.clamp(mel_output, min=self.mel_extractor.clamp))
            
            mel = log_mel_spec
            
            # Pad mel to be divisible by 32
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = torch.nn.functional.pad(mel, (0, n_pad), mode="constant")
                
            # E2E model expects (B, n_mels, T_frames)
            hidden = self.model(mel)
            
            # Slice back to original length
            return hidden[:, :n_frames]

    model = RMVPEWrapper(rmvpe_instance).to(device)
    model.eval()
    
    # Dummy input
    # 16k sample rate, 1 second audio
    dummy_audio = torch.randn(1, 16000).to(device)
    
    output_path = os.path.join(output_dir, "rmvpe.onnx")
    
    torch.onnx.export(
        model,
        (dummy_audio,),
        output_path,
        input_names=["audio"],
        output_names=["hidden"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "time"},
            "hidden": {0: "batch_size", 1: "time"},
        },
        opset_version=17 # Required for STFT
    )
    logger.info(f"RMVPE exported to {output_path}")

if __name__ == "__main__":
    main()
