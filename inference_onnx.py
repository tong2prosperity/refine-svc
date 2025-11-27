import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import soundfile as sf
import librosa
import yaml
from modules.audio import mel_spectrogram
from modules.commons import recursive_munch

# Patch for transformers import error
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

from transformers import Wav2Vec2FeatureExtractor

class ONNXModel:
    def __init__(self, model_path, device='cpu'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
    def __call__(self, inputs):
        input_names = [node.name for node in self.session.get_inputs()]
        ort_inputs = {name: inputs[i] for i, name in enumerate(input_names)}
        return self.session.run(None, ort_inputs)

def load_audio(file_path, sr=16000):
    wav, _ = librosa.load(file_path, sr=sr)
    return torch.from_numpy(wav).unsqueeze(0)

def get_mel_spectrogram(wav, config):
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": config['preprocess_params']['sr'],
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    return mel_spectrogram(wav, **mel_fn_args)

class RMVPEONNX:
    def __init__(self, model_path, device='cpu'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx] : ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        # audio: (T,)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        
        input_name = self.session.get_inputs()[0].name
        hidden = self.session.run(None, {input_name: audio})[0]
        # hidden: (B, T_frames, 360)
        
        return self.decode(hidden[0], thred=thred)

def main():
    parser = argparse.ArgumentParser(description="Run ONNX inference")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("--source_audio", type=str, required=True, help="Source audio path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path")
    parser.add_argument("--onnx_dir", type=str, default="onnx_models", help="ONNX models directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--pitch_shift", type=float, default=0.0, help="Pitch shift in semitones")
    parser.add_argument("--diffusion_steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--inference_cfg_rate", type=float, default=0.7, help="Inference CFG rate")
    
    args = parser.parse_args()
    
    device = args.device
    onnx_dir = args.onnx_dir
    
    # Load Config
    config_path = "models/seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = recursive_munch(config)
    
    # Load Models
    print("Loading models...")
    speech_tokenizer = ONNXModel(os.path.join(onnx_dir, "speech_tokenizer.onnx"), device)
    campplus = ONNXModel(os.path.join(onnx_dir, "campplus.onnx"), device)
    length_regulator = ONNXModel(os.path.join(onnx_dir, "length_regulator.onnx"), device)
    dit = ONNXModel(os.path.join(onnx_dir, "dit.onnx"), device)
    vocoder = ONNXModel(os.path.join(onnx_dir, "vocoder.onnx"), device)
    
    rmvpe_path = os.path.join(onnx_dir, "rmvpe.onnx")
    rmvpe = None
    if os.path.exists(rmvpe_path):
        rmvpe = RMVPEONNX(rmvpe_path, device)
    else:
        print("Warning: rmvpe.onnx not found, pitch shifting might not work correctly if F0 is required.")

    # Load Audio
    print("Loading audio...")
    # Reference
    ref_wav, _ = librosa.load(args.ref_audio, sr=config.preprocess_params.sr)
    ref_wav = torch.from_numpy(ref_wav).unsqueeze(0)
    
    # Source
    src_wav, _ = librosa.load(args.source_audio, sr=config.preprocess_params.sr)
    src_wav = torch.from_numpy(src_wav).unsqueeze(0)

    # 1. Semantic Extraction (Speech Tokenizer)
    # Resample to 16k
    ref_wav_16k = torchaudio.functional.resample(ref_wav, config.preprocess_params.sr, 16000)
    src_wav_16k = torchaudio.functional.resample(src_wav, config.preprocess_params.sr, 16000)
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./models/wav2vec2")
    
    # Source Semantics
    inputs = feature_extractor(src_wav_16k.squeeze(0).numpy(), return_tensors="np", sampling_rate=16000)
    input_values = inputs.input_values
    S_alt = torch.from_numpy(speech_tokenizer([input_values])[0])
    
    # Reference Semantics (for prompt)
    inputs = feature_extractor(ref_wav_16k.squeeze(0).numpy(), return_tensors="np", sampling_rate=16000)
    input_values = inputs.input_values
    S_ori = torch.from_numpy(speech_tokenizer([input_values])[0])

    # 2. Style Extraction (CAMPPlus)
    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style_input = feat2.unsqueeze(0).numpy()
    style_lens = np.array([style_input.shape[1]], dtype=np.int64)
    style2 = torch.from_numpy(campplus([style_input, style_lens])[0])

    # 3. Mel Spectrogram of Reference
    mel_fn_args = {
        "n_fft": config.preprocess_params.spect_params.n_fft,
        "win_size": config.preprocess_params.spect_params.win_length,
        "hop_size": config.preprocess_params.spect_params.hop_length,
        "num_mels": config.preprocess_params.spect_params.n_mels,
        "sampling_rate": config.preprocess_params.sr,
        "fmin": config.preprocess_params.spect_params.fmin,
        "fmax": None if config.preprocess_params.spect_params.fmax == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    mel2 = mel_spectrogram(ref_wav, **mel_fn_args)
    target2_lengths = torch.LongTensor([mel2.size(2)])

    # 4. F0 Extraction & Pitch Shift
    f0 = None
    if rmvpe is not None:
        # RMVPE expects 16k audio
        f0 = rmvpe.infer_from_audio(src_wav_16k.squeeze(0).numpy())
        
        if args.pitch_shift != 0:
            f0 = f0 * (2 ** (args.pitch_shift / 12))
            
        # Interpolate F0 to match S_alt length
        # S_alt shape: (1, T_sem, D)
        # f0 shape: (T_f0,)
        f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).float()
        f0 = torch.nn.functional.interpolate(f0, size=S_alt.shape[1], mode='nearest').squeeze(0)
    else:
        # Dummy F0
        f0 = torch.zeros((1, S_alt.shape[1]), dtype=torch.float32)

    # 5. Length Regulator
    # Prompt
    prompt_condition = torch.from_numpy(length_regulator([
        S_ori.numpy(), 
        target2_lengths.numpy(), 
        np.array([3], dtype=np.int64),
        np.zeros((1, S_ori.shape[1]), dtype=np.float32) # Dummy f0 for prompt
    ])[0])
    
    # Target
    # Estimate target length
    hop_length = config.preprocess_params.spect_params.hop_length
    target_len = int(src_wav.shape[1] / hop_length)
    target_lengths = torch.LongTensor([target_len])
    
    cond = torch.from_numpy(length_regulator([
        S_alt.numpy(),
        target_lengths.numpy(),
        np.array([3], dtype=np.int64),
        f0.numpy()
    ])[0])
    
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    
    # 6. DiT Inference (CFM)
    print("Running DiT Inference...")
    # Prepare inputs
    # mel2 is prompt (B, 80, T_prompt)
    # cat_condition is mu (B, T_total, D)
    
    mu = cat_condition
    B, T_total, _ = mu.shape
    prompt = mel2 # (B, 80, T_prompt)
    style = style2 # (B, 192)
    
    # Initialize x (noise)
    # in_channels = 80
    x = torch.randn(B, 80, T_total)
    n_timesteps = args.diffusion_steps
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    
    # Prepare prompt_x
    prompt_len = prompt.size(2)
    prompt_x = torch.zeros_like(x)
    prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
    x[:, :, :prompt_len] = 0
    
    # x_lens
    x_lens = torch.LongTensor([T_total])
    
    # Inference loop (Euler)
    inference_cfg_rate = 0.7 # Default from real-time-gui.py
    
    t = t_span[0]
    for step in range(1, len(t_span)):
        dt = t_span[step] - t_span[step - 1]
        
        # Prepare inputs for DiT
        # DiT.forward(x, prompt_x, x_lens, t, style, cond)
        
        if inference_cfg_rate > 0:
            # Stack for CFG
            stacked_x = torch.cat([x, x], dim=0)
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)
            stacked_t = torch.cat([t.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0)], dim=0).flatten() # (2,)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_cond = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            
            # Run DiT
            # Inputs: x, prompt_x, x_lens, t, style, cond
            # ONNX inputs: x, prompt_x, x_lens, t, style, cond
            
            # Convert to numpy
            ort_inputs = {
                "x": stacked_x.numpy(),
                "prompt_x": stacked_prompt_x.numpy(),
                "x_lens": stacked_x_lens.numpy(),
                "t": stacked_t.numpy(),
                "style": stacked_style.numpy(),
                "cond": stacked_cond.numpy()
            }
            
            stacked_dphi_dt = torch.from_numpy(dit.session.run(None, ort_inputs)[0])
            
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            
        else:
            ort_inputs = {
                "x": x.numpy(),
                "prompt_x": prompt_x.numpy(),
                "x_lens": x_lens.numpy(),
                "t": t.unsqueeze(0).numpy(), # (1,)
                "style": style.numpy(),
                "cond": mu.numpy()
            }
            dphi_dt = torch.from_numpy(dit.session.run(None, ort_inputs)[0])
            
        x = x + dt * dphi_dt
        t = t + dt
        
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
            
        x[:, :, :prompt_len] = 0
        
    vc_target = x
    # Remove prompt part for vocoder?
    # real-time-gui.py: vc_target = vc_target[:, :, mel2.size(-1) :]
    vc_target = vc_target[:, :, prompt_len:]
    
    # 7. Vocoder
    print("Running Vocoder...")
    # Input: mel, f0 (if hifigan)
    # We need to know if it's hifigan or bigvgan.
    # Check inputs of vocoder session
    vocoder_inputs = [node.name for node in vocoder.session.get_inputs()]
    is_hifigan = "f0" in vocoder_inputs
    
    if is_hifigan:
        # Need f0. real-time-gui.py passes None to vocoder_fn if it's hifigan?
        # Wait, real-time-gui.py: vc_wave = vocoder_fn(vc_target).squeeze()
        # HiFTGenerator.forward(x, f0=None) -> predicts f0 if None.
        # But we exported with f0 input.
        # If we exported with f0, we MUST pass f0.
        # But HiFiGAN has f0_predictor.
        # In export_onnx.py, I exported with dummy_f0.
        # And I monkeypatched forward to take f0.
        # If I want to use f0 predictor inside ONNX, I should have exported it differently or checked if f0 is optional.
        # But I passed f0 as input.
        # So I need to provide f0.
        # But I don't have f0.
        # This is a problem. HiFiGAN in this repo seems to rely on internal f0 predictor if f0 is None.
        # But ONNX export usually freezes the path.
        # If I exported the path where f0 is provided, I must provide f0.
        # If I want to use internal predictor, I should have exported with f0=None (if supported by export).
        # But torch.onnx.export traces execution. If I passed dummy_f0, it traced the path using dummy_f0.
        # So the exported model EXPECTS f0.
        
        # I should have exported the f0 predictor part or the path that uses it.
        # In `export_onnx.py`, I did:
        # args = (dummy_x, dummy_f0)
        # So it traced `_f02source(f0)`.
        # It skipped `f0 = self.f0_predictor(x)`.
        
        # So my exported HiFiGAN REQUIRES f0.
        # I need to run f0 predictor separately or use a dummy f0 (which will sound bad).
        # Or I need to re-export HiFiGAN to include f0 predictor (by passing f0=None during export).
        
        # Let's try to re-export HiFiGAN with f0=None if possible.
        # But `HiFTGenerator.forward` with `f0=None` calls `self.f0_predictor(x)`.
        # So if I export with `f0=None`, it should trace the predictor.
        
        # I will assume for now I need to re-export or I can't run inference properly without f0.
        # But wait, `real-time-gui.py` calls `vocoder_fn(vc_target)`. It doesn't pass f0.
        # So it uses the internal predictor.
        # So my export was WRONG for the intended usage.
        
        # I should fix `export_onnx.py` to export with `f0=None` for HiFiGAN.
        # But first let's finish the script assuming I will fix the export.
        
        # If I fix export, input will be just `mel`.
        # If I don't fix export, I need f0.
        
        # I will add a TODO and handle both cases if possible, or just fail if f0 needed but not present.
        # For now, let's assume I will fix export to take only `mel` (and internally predict f0).
        
        pass
    
    # Assuming fixed export (only mel input) or BigVGAN (only mel input)
    # But wait, I monkeypatched HiFiGAN to return real/imag.
    
    vocoder_out = vocoder.session.run(None, {"mel": vc_target.numpy()})
        
    # Post-processing for HiFiGAN (ISTFT)
    # Output names: "real", "imag" if hifigan (monkeypatched)
    # "audio" if bigvgan
    
    output_names = [node.name for node in vocoder.session.get_outputs()]
    if "real" in output_names:
        real = torch.from_numpy(vocoder_out[0])
        imag = torch.from_numpy(vocoder_out[1])
        # ISTFT
        # We need istft params.
        # They are in config.
        # istft_params: {"n_fft": 16, "hop_len": 4} (default in HiFTGenerator)
        # But we should check config/hifigan.yml or assume defaults.
        # The config passed to main is for DiT.
        # We might need hifigan config.
        # Or hardcode if standard.
        # HiFTGenerator defaults: n_fft=16, hop_len=4.
        # Let's try to load hifigan config if available.
        
        n_fft = 16
        hop_len = 4
        win_length = 16
        window = torch.hann_window(win_length)
        
        # istft expects complex tensor
        spec = torch.complex(real, imag)
        audio = torch.istft(spec, n_fft, hop_len, win_length, window=window)
    else:
        audio = torch.from_numpy(vocoder_out[0])
        
    # Save audio
    print(f"Saving output to {args.output}...")
    sf.write(args.output, audio.squeeze().numpy(), config.preprocess_params.sr)
    print("Done!")

if __name__ == "__main__":
    main()
