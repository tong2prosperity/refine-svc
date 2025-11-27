import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as tat
import yaml
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from transformers import Wav2Vec2FeatureExtractor

from modules.audio import mel_spectrogram
from modules.rmvpe import RMVPE
from utils.audio_assets import FileMap

logger = logging.getLogger(__name__)


class ONNXModel:
    def __init__(self, model_path, device='cpu'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

    def __call__(self, inputs):
        input_names = [node.name for node in self.session.get_inputs()]
        ort_inputs = {name: inputs[i] for i, name in enumerate(input_names)}
        return self.session.run(None, ort_inputs)


@dataclass
class RefineONNXConfig:
    diffusion_steps: int = 10
    inference_cfg_rate: float = 0.7
    max_prompt_length: float = 3.0
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time_ce: float = 2.5
    extra_time: float = 0.5
    extra_time_right: float = 2.0
    pitch_shift: float = 0.0  # Semitones

    def apply_overrides(self, overrides: Optional[Dict[str, float]]) -> None:
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RefineONNXSession:
    def __init__(
        self,
        *,
        onnx_models: Dict[str, ONNXModel],
        rmvpe: RMVPE,
        feature_extractor: Wav2Vec2FeatureExtractor,
        device: str,
        reference_audio_path: str,
        source_sample_rate: int,
        config: RefineONNXConfig,
        model_config: Dict,
    ) -> None:
        if config.extra_time_ce < config.extra_time:
            raise ValueError("extra_time_ce must be greater than extra_time")
        self.onnx_models = onnx_models
        self.rmvpe = rmvpe
        self.feature_extractor = feature_extractor
        self.device = device
        self.reference_audio_path = str(Path(reference_audio_path).expanduser().resolve())
        self.source_sample_rate = source_sample_rate
        self.config = config
        self.model_config = model_config
        
        self.model_sr = self.model_config['preprocess_params']['sr']
        self.stream_sample_rate = self.model_sr
        self.channels = 1
        
        # Load and process reference audio
        self.reference_wav, _ = librosa.load(
            self.reference_audio_path, sr=self.model_sr
        )
        self.reference_wav = torch.from_numpy(self.reference_wav).unsqueeze(0)
        
        # Pre-calculate reference features
        self._prepare_reference_features()
        
        self._prepare_buffers()
        self.input_float_buffer = np.array([], dtype=np.float32)
        self.silence_blocks_remaining: Optional[int] = None
        self._flushed = False
        
        # VAD state (simple energy based or reuse existing if needed, for now simple pass-through or always active)
        # In this implementation we will assume always active or simple VAD if integrated.
        # For parity with RealTimeVCService, we should ideally use a VAD.
        # But to keep ONNX service self-contained and simple first, we might skip complex VAD or use a simple one.
        # The original service uses `funasr` VAD. We can reuse it if we want, but let's stick to core logic first.
        self.vad_speech_detected = True 

    def _prepare_reference_features(self):
        # 1. Semantic Extraction (Speech Tokenizer)
        # Resample to 16k for semantic extraction
        ref_wav_16k = torchaudio.functional.resample(self.reference_wav, self.model_sr, 16000)
        
        inputs = self.feature_extractor(ref_wav_16k.squeeze(0).numpy(), return_tensors="np", sampling_rate=16000)
        input_values = inputs.input_values
        self.S_ori = torch.from_numpy(self.onnx_models['speech_tokenizer']([input_values])[0])

        # 2. Style Extraction (CAMPPlus)
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style_input = feat2.unsqueeze(0).numpy()
        style_lens = np.array([style_input.shape[1]], dtype=np.int64)
        self.style2 = torch.from_numpy(self.onnx_models['campplus']([style_input, style_lens])[0])

        # 3. Mel Spectrogram of Reference
        self.mel2 = self._get_mel_spectrogram(self.reference_wav)
        self.target2_lengths = torch.LongTensor([self.mel2.size(2)])

        # 4. Length Regulator (Prompt)
        # Note: ONNX export for length regulator expects: x, ylens, n_quantizers, f0
        # We pass dummy f0 as it's not used for prompt usually or we don't shift prompt pitch
        self.prompt_condition = torch.from_numpy(self.onnx_models['length_regulator']([
            self.S_ori.numpy(),
            self.target2_lengths.numpy(),
            np.array([3], dtype=np.int64),
            np.zeros((1, self.S_ori.shape[1]), dtype=np.float32)
        ])[0])

    def _get_mel_spectrogram(self, wav):
        mel_fn_args = {
            "n_fft": self.model_config['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.model_config['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.model_config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.model_config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.model_config['preprocess_params']['sr'],
            "fmin": self.model_config['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.model_config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        return mel_spectrogram(wav, **mel_fn_args)

    @property
    def output_sample_rate(self) -> int:
        return self.stream_sample_rate

    def _prepare_buffers(self) -> None:
        self.zc = max(self.stream_sample_rate // 50, 1)
        self.block_frame = (
            int(np.round(self.config.block_time * self.stream_sample_rate / self.zc))
            * self.zc
        )
        if self.block_frame == 0:
            self.block_frame = self.zc
        
        self.crossfade_frame = (
            int(np.round(self.config.crossfade_time * self.stream_sample_rate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        if self.sola_buffer_frame == 0:
            self.sola_buffer_frame = self.zc
        self.sola_search_frame = self.zc
        
        self.extra_frame = (
            int(np.round(self.config.extra_time_ce * self.stream_sample_rate / self.zc))
            * self.zc
        )
        self.extra_frame_right = (
            int(np.round(self.config.extra_time_right * self.stream_sample_rate / self.zc))
            * self.zc
        )
        
        total_length = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right
        )
        
        self.input_wav = torch.zeros(
            total_length, device='cpu', dtype=torch.float32
        )
        
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device='cpu', dtype=torch.float32
        )
        self.sola_buffer_has_data = False
        
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device='cpu',
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.ones_kernel = torch.ones(
            1, 1, self.sola_buffer_frame, device='cpu', dtype=torch.float32
        )
        
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

    def _ensure_block_frame(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() < self.block_frame:
            padding = torch.zeros(
                self.block_frame - tensor.numel(),
                device='cpu',
                dtype=torch.float32,
            )
            tensor = torch.cat([tensor, padding], dim=0)
        elif tensor.numel() > self.block_frame:
            tensor = tensor[-self.block_frame :]
        return tensor.contiguous()

    def _shift_and_append(self, tensor: torch.Tensor) -> None:
        if tensor.numel() != self.block_frame:
            raise ValueError("Tensor length must equal block_frame after padding")
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-self.block_frame :] = tensor

    def _apply_sola(self, infer_wav: torch.Tensor) -> torch.Tensor:
        required_length = self.sola_buffer_frame + self.sola_search_frame
        if infer_wav.numel() < required_length:
            infer_wav = F.pad(
                infer_wav,
                (0, required_length - infer_wav.numel()),
                value=0.0,
            )
        conv_input = infer_wav[:required_length].unsqueeze(0).unsqueeze(0)
        cor_nom = F.conv1d(conv_input, self.sola_buffer.unsqueeze(0).unsqueeze(0))
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2, self.ones_kernel) + 1e-8
        )
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = torch.argmax(tensor).item()
        infer_wav = infer_wav[sola_offset:]
        if infer_wav.numel() < self.block_frame + self.sola_buffer_frame:
            infer_wav = F.pad(
                infer_wav,
                (
                    0,
                    self.block_frame
                    + self.sola_buffer_frame
                    - infer_wav.numel(),
                ),
                value=0.0,
            )
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (
            self.sola_buffer * self.fade_out_window
        )
        self.sola_buffer = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ].clone()
        self.sola_buffer_has_data = True
        return infer_wav[: self.block_frame]

    def _process_block(self, tensor: torch.Tensor) -> bytes:
        tensor = self._ensure_block_frame(tensor)
        self._shift_and_append(tensor)
        
        # Inference
        infer_wav = self._run_inference()
        
        if infer_wav is None:
            return b""
            
        infer_wav = infer_wav.clamp_(-1.0, 1.0)
        output = infer_wav.numpy()
        output_int16 = (output * 32767.0).astype(np.int16)
        return output_int16.tobytes()

    def _run_inference(self) -> Optional[torch.Tensor]:
        # 1. Prepare input
        # input_wav is at model_sr
        # We need to resample to 16k for semantic extraction and F0
        input_wav_res = self.input_wav.clone()
        
        # Resample to 16k
        input_wav_16k = torchaudio.functional.resample(input_wav_res.unsqueeze(0), self.model_sr, 16000)
        
        # 2. Semantic Extraction
        inputs = self.feature_extractor(input_wav_16k.squeeze(0).numpy(), return_tensors="np", sampling_rate=16000)
        input_values = inputs.input_values
        S_alt = torch.from_numpy(self.onnx_models['speech_tokenizer']([input_values])[0])
        
        # Crop S_alt to match the processing window logic
        # In real-time-gui.py: S_alt = S_alt[:, ce_dit_frame_difference:]
        # ce_dit_frame_difference = int(ce_dit_difference * 50)
        # ce_dit_difference is config.extra_time_ce - config.extra_time
        ce_dit_difference = self.config.extra_time_ce - self.config.extra_time
        ce_dit_frame_difference = int(ce_dit_difference * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        
        # 3. F0 Estimation & Pitch Shift
        # We need F0 for the length regulator if we want to support pitch shifting properly
        # Or if the model was trained/exported with F0 condition.
        # Assuming we want to support pitch shifting:
        f0 = self.rmvpe.infer_from_audio(input_wav_16k.squeeze(0), thred=0.03)
        
        # Apply pitch shift
        if self.config.pitch_shift != 0:
            f0 = f0 * (2 ** (self.config.pitch_shift / 12))
            
        # Resize f0 to match S_alt length
        # S_alt shape (1, T, D)
        # f0 shape (T_audio_16k // hop,) -> RMVPE hop is 160 (10ms)
        # S_alt is 50Hz (20ms)
        # We need to interpolate f0 to S_alt length
        f0 = torch.from_numpy(f0).unsqueeze(0).unsqueeze(0).float() # (1, 1, T_f0)
        f0 = F.interpolate(f0, size=S_alt.shape[1], mode='nearest').squeeze(0) # (1, T_sem)
        
        # 4. Length Regulator (Target)
        hop_length = self.model_config['preprocess_params']['spect_params']['hop_length']
        target_len = int((self.skip_head + self.return_length + self.skip_tail - ce_dit_frame_difference) / 50 * self.model_sr // hop_length)
        target_lengths = torch.LongTensor([target_len])
        
        cond = torch.from_numpy(self.onnx_models['length_regulator']([
            S_alt.numpy(),
            target_lengths.numpy(),
            np.array([3], dtype=np.int64),
            f0.numpy()
        ])[0])
        
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
        
        # 5. DiT Inference
        # Prepare inputs
        mu = cat_condition
        B, T_total, _ = mu.shape
        prompt = self.mel2
        style = self.style2
        
        x = torch.randn(B, 80, T_total)
        n_timesteps = int(self.config.diffusion_steps)
        t_span = torch.linspace(0, 1, n_timesteps + 1)
        
        prompt_len = prompt.size(2)
        prompt_x = torch.zeros_like(x)
        prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
        x[:, :, :prompt_len] = 0
        x_lens = torch.LongTensor([T_total])
        
        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            
            if self.config.inference_cfg_rate > 0:
                stacked_x = torch.cat([x, x], dim=0)
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0)], dim=0).flatten()
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_cond = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                
                ort_inputs = {
                    "x": stacked_x.numpy(),
                    "prompt_x": stacked_prompt_x.numpy(),
                    "x_lens": stacked_x_lens.numpy(),
                    "t": stacked_t.numpy(),
                    "style": stacked_style.numpy(),
                    "cond": stacked_cond.numpy()
                }
                
                stacked_dphi_dt = torch.from_numpy(self.onnx_models['dit'](list(ort_inputs.values()))[0])
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + self.config.inference_cfg_rate) * dphi_dt - self.config.inference_cfg_rate * cfg_dphi_dt
            else:
                ort_inputs = {
                    "x": x.numpy(),
                    "prompt_x": prompt_x.numpy(),
                    "x_lens": x_lens.numpy(),
                    "t": t.unsqueeze(0).numpy(),
                    "style": style.numpy(),
                    "cond": mu.numpy()
                }
                dphi_dt = torch.from_numpy(self.onnx_models['dit'](list(ort_inputs.values()))[0])
                
            x = x + dt * dphi_dt
            t = t + dt
            
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
                
            x[:, :, :prompt_len] = 0
            
        vc_target = x[:, :, prompt_len:]
        
        # 6. Vocoder
        # Check if HiFiGAN (needs f0) or BigVGAN
        # For now assuming HiFiGAN exported with f0 input, or we need to handle it.
        # In the plan we decided to use RMVPE f0.
        # If vocoder expects f0, we need to provide it.
        # The f0 we have is aligned to semantic frames (50Hz).
        # Vocoder expects f0 aligned to mel frames?
        # HiFiGAN usually takes f0 interpolated to audio length or mel length.
        # Let's check what we passed to length regulator: f0 aligned to S_alt.
        
        # We need f0 for the TARGET part only (vc_target).
        # The f0 we computed earlier covers the whole input window.
        # We need to slice it to match vc_target.
        # But wait, vc_target corresponds to the generated mel.
        # The generated mel corresponds to S_alt (which was cropped).
        
        # Let's check vocoder input names
        vocoder_inputs = [node.name for node in self.onnx_models['vocoder'].session.get_inputs()]
        vocoder_args = {"mel": vc_target.numpy()}
        
        if "f0" in vocoder_inputs:
            # We need to provide f0.
            # We have f0 from RMVPE (1, T_sem).
            # We need to interpolate it to match vc_target time axis.
            # vc_target shape (1, 80, T_mel)
            target_mel_len = vc_target.shape[2]
            
            # Interpolate f0 to target_mel_len
            f0_for_vocoder = F.interpolate(f0.unsqueeze(0), size=target_mel_len, mode='nearest').squeeze(0)
            vocoder_args["f0"] = f0_for_vocoder.numpy()
            
        vocoder_out = self.onnx_models['vocoder'](list(vocoder_args.values()))
        
        # Handle HiFiGAN output (real/imag) or BigVGAN (audio)
        output_names = [node.name for node in self.onnx_models['vocoder'].session.get_outputs()]
        if "real" in output_names:
            real = torch.from_numpy(vocoder_out[0])
            imag = torch.from_numpy(vocoder_out[1])
            # ISTFT
            n_fft = 16
            hop_len = 4
            win_length = 16
            window = torch.hann_window(win_length)
            spec = torch.complex(real, imag)
            audio = torch.istft(spec, n_fft, hop_len, win_length, window=window)
        else:
            audio = torch.from_numpy(vocoder_out[0])
            
        # Apply SOLA
        return self._apply_sola(audio.squeeze(0))

    def process_pcm(self, pcm_bytes: bytes) -> Optional[bytes]:
        if not pcm_bytes:
            return None
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return None
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        if self.source_sample_rate != self.stream_sample_rate:
            audio_stream = librosa.resample(
                audio_float,
                orig_sr=self.source_sample_rate,
                target_sr=self.stream_sample_rate,
            ).astype(np.float32)
        else:
            audio_stream = audio_float
            
        self.input_float_buffer = np.concatenate([self.input_float_buffer, audio_stream])
        
        output_bytes = b""
        while len(self.input_float_buffer) >= self.block_frame:
            chunk = self.input_float_buffer[:self.block_frame]
            self.input_float_buffer = self.input_float_buffer[self.block_frame:]
            tensor = torch.from_numpy(chunk)
            output_bytes += self._process_block(tensor)
            
        return output_bytes if output_bytes else None

    def flush(self) -> Optional[bytes]:
        if self._flushed:
            return None

        flush_output = b""

        if len(self.input_float_buffer) > 0:
            chunk = self.input_float_buffer
            self.input_float_buffer = np.array([], dtype=np.float32)
            tensor = torch.from_numpy(chunk)
            flush_output += self._process_block(tensor)
            return flush_output

        if self.silence_blocks_remaining is None:
            self.silence_blocks_remaining = math.ceil(self.config.extra_time_right / self.config.block_time)

        if self.silence_blocks_remaining > 0:
            self.silence_blocks_remaining -= 1
            zeros = torch.zeros(self.block_frame, dtype=torch.float32)
            flush_output += self._process_block(zeros)
            return flush_output

        if self.sola_buffer_has_data and self.sola_buffer.numel() > 0:
            sola_audio = self.sola_buffer.clamp_(-1.0, 1.0)
            flush_output += (sola_audio.numpy() * 32767.0).astype(np.int16).tobytes()

        self._flushed = True
        return flush_output if flush_output else None


class RefineONNXService:
    def __init__(
        self,
        voice_root: str = "examples/reference",
        onnx_dir: str = "onnx_models",
        config_path: str = "models/seed-vc/config_dit_mel_seed_uvit_xlsr_tiny.yml",
        device: str = "cpu"
    ) -> None:
        self.device = device
        self.onnx_dir = onnx_dir
        self.config_path = config_path
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
            
        # Load ONNX models
        logger.info("Loading ONNX models...")
        self.onnx_models = {
            "speech_tokenizer": ONNXModel(os.path.join(onnx_dir, "speech_tokenizer.onnx"), device),
            "campplus": ONNXModel(os.path.join(onnx_dir, "campplus.onnx"), device),
            "length_regulator": ONNXModel(os.path.join(onnx_dir, "length_regulator.onnx"), device),
            "dit": ONNXModel(os.path.join(onnx_dir, "dit.onnx"), device),
            "vocoder": ONNXModel(os.path.join(onnx_dir, "vocoder.onnx"), device),
        }
        
        # Load RMVPE
        logger.info("Loading RMVPE model...")
        self.rmvpe = RMVPE(
            os.path.join("models/rmvpe", "rmvpe.pt"), # Assuming standard path or env var
            is_half=False,
            device=device
        )
        
        # Load Feature Extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./models/wav2vec2")
        
        self.file_map = FileMap(voice_root, file_extensions=[".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"])

    def list_voices(self) -> List[Dict[str, str]]:
        return [
            {"id": name, "title": name}
            for name in sorted(self.file_map.get_filenames())
        ]

    def create_session(
        self,
        voice_id: str,
        source_sample_rate: int,
        overrides: Optional[Dict[str, float]] = None,
    ) -> RefineONNXSession:
        if voice_id not in self.file_map:
            raise KeyError(f"Unknown voice id: {voice_id}")
        reference_path = self.file_map[voice_id]
        config = RefineONNXConfig()
        config.apply_overrides(overrides)
        
        return RefineONNXSession(
            onnx_models=self.onnx_models,
            rmvpe=self.rmvpe,
            feature_extractor=self.feature_extractor,
            device=self.device,
            reference_audio_path=reference_path,
            source_sample_rate=source_sample_rate,
            config=config,
            model_config=self.model_config,
        )
