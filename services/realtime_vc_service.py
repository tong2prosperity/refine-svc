import asyncio
import importlib.util
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from fastapi import WebSocket
from funasr import AutoModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

from utils.audio_assets import FileMap

logger = logging.getLogger(__name__)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_REALTIME_GUI_MODULE = None


def _load_realtime_gui_module():
    global _REALTIME_GUI_MODULE
    if _REALTIME_GUI_MODULE is not None:
        return _REALTIME_GUI_MODULE
    module_path = Path(__file__).resolve().parent.parent / "real-time-gui.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find real-time-gui.py at {module_path}")
    spec = importlib.util.spec_from_file_location("realtime_gui_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load module specification for real-time-gui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REALTIME_GUI_MODULE = module
    return module


@dataclass
class RealTimeVCConfig:
    diffusion_steps: int = 10
    inference_cfg_rate: float = 0.7
    max_prompt_length: float = 3.0
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time_ce: float = 2.5
    extra_time: float = 0.5
    extra_time_right: float = 2.0

    def apply_overrides(self, overrides: Optional[Dict[str, float]]) -> None:
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RealTimeVCSession:
    def __init__(
        self,
        *,
        model_set,
        device: torch.device,
        vad_model: AutoModel,
        reference_audio_path: str,
        source_sample_rate: int,
        config: RealTimeVCConfig,
        module,
    ) -> None:
        if config.extra_time_ce < config.extra_time:
            raise ValueError("extra_time_ce must be greater than extra_time")
        self.module = module
        self.model_set = model_set
        self.device = device
        self.vad_model = vad_model
        self.reference_audio_path = str(Path(reference_audio_path).expanduser().resolve())
        self.source_sample_rate = source_sample_rate
        self.config = config
        self.model_sr = self.model_set[-1]["sampling_rate"]
        self.stream_sample_rate = self.model_sr
        self.channels = 1
        self._reset_module_cache()
        self.reference_wav, _ = librosa.load(
            self.reference_audio_path, sr=self.model_sr
        )
        self.reference_wav = self.reference_wav.astype(np.float32)
        self._prepare_buffers()

    @property
    def output_sample_rate(self) -> int:
        return self.stream_sample_rate

    def _reset_module_cache(self) -> None:
        self.module.prompt_condition = None
        self.module.mel2 = None
        self.module.style2 = None
        self.module.reference_wav_name = ""
        self.module.prompt_len = 3
        self.module.ce_dit_difference = 2.0

    def _prepare_buffers(self) -> None:
        self.zc = max(self.stream_sample_rate // 50, 1)
        self.block_frame = (
            int(
                np.round(
                    self.config.block_time * self.stream_sample_rate / self.zc
                )
            )
            * self.zc
        )
        if self.block_frame == 0:
            self.block_frame = self.zc
        self.block_frame_16k = max(320 * self.block_frame // self.zc, 320)
        self.crossfade_frame = (
            int(
                np.round(
                    self.config.crossfade_time * self.stream_sample_rate / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        if self.sola_buffer_frame == 0:
            self.sola_buffer_frame = self.zc
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.config.extra_time_ce * self.stream_sample_rate / self.zc
                )
            )
            * self.zc
        )
        self.extra_frame_right = (
            int(
                np.round(
                    self.config.extra_time_right * self.stream_sample_rate / self.zc
                )
            )
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
            total_length, device=self.device, dtype=torch.float32
        )
        self.input_wav_res = torch.zeros(
            max(320 * total_length // self.zc, self.block_frame_16k),
            device=self.device,
            dtype=torch.float32,
        )
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.device, dtype=torch.float32
        )
        self.sola_buffer_has_data = False  # Track if the buffer contains actual audio data
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.ones_kernel = torch.ones(
            1, 1, self.sola_buffer_frame, device=self.device, dtype=torch.float32
        )
        self.resampler = tat.Resample(
            orig_freq=self.stream_sample_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.device)
        if self.model_sr != self.stream_sample_rate:
            self.resampler2 = tat.Resample(
                orig_freq=self.model_sr,
                new_freq=self.stream_sample_rate,
                dtype=torch.float32,
            ).to(self.device)
        else:
            self.resampler2 = None
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.vad_cache: Dict[str, np.ndarray] = {}
        self.vad_chunk_size = min(500, int(1000 * self.config.block_time))
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        self._flushed = False

    def _ensure_block_frame(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() < self.block_frame:
            padding = torch.zeros(
                self.block_frame - tensor.numel(),
                device=self.device,
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

    def _update_resampled_buffer(self) -> None:
        resampled = self.resampler(self.input_wav.unsqueeze(0)).squeeze(0)
        if resampled.numel() < self.input_wav_res.numel():
            pad = torch.zeros(
                self.input_wav_res.numel() - resampled.numel(),
                device=self.device,
                dtype=torch.float32,
            )
            resampled = torch.cat([pad, resampled], dim=0)
        self.input_wav_res = resampled[-self.input_wav_res.numel() :]

    def _update_vad(self, indata_16k: np.ndarray = None) -> None:
        # For debugging purposes, always detect speech
        self.vad_speech_detected = True
        self.set_speech_detected_false_at_end_flag = False
        # Original VAD code for reference
        # res = self.vad_model.generate(
        #     input=indata_16k,
        #     cache=self.vad_cache,
        #     is_final=False,
        #     chunk_size=self.vad_chunk_size,
        # )
        # res_value = res[0]["value"]
        # if len(res_value) % 2 == 1 and not self.vad_speech_detected:
        #     self.vad_speech_detected = True
        # elif len(res_value) % 2 == 1 and self.vad_speech_detected:
        #     self.set_speech_detected_false_at_end_flag = True

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
        self.sola_buffer_has_data = True  # Mark that the buffer now contains real audio data
        return infer_wav[: self.block_frame]

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
        tensor = torch.from_numpy(audio_stream).to(self.device)
        tensor = self._ensure_block_frame(tensor)
        self._shift_and_append(tensor)
        indata_16k = librosa.resample(
            audio_float,
            orig_sr=self.source_sample_rate,
            target_sr=16000,
        ).astype(np.float32)
        self._update_vad(indata_16k)
        self._update_resampled_buffer()
        infer_wav = self._run_inference()
        if infer_wav is None:
            return None
        if self.set_speech_detected_false_at_end_flag:
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False
        infer_wav = infer_wav.clamp_(-1.0, 1.0)
        output = infer_wav.cpu().numpy()
        output_int16 = (output * 32767.0).astype(np.int16)
        return output_int16.tobytes()

    def _run_inference(self, force: bool = False) -> Optional[torch.Tensor]:
        if not force and not self.vad_speech_detected:
            return torch.zeros(
                self.block_frame, device=self.device, dtype=torch.float32
            )
        infer_wav = self.module.custom_infer(
            self.model_set,
            self.reference_wav,
            self.reference_audio_path,
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.skip_tail,
            self.return_length,
            int(self.config.diffusion_steps),
            self.config.inference_cfg_rate,
            self.config.max_prompt_length,
            self.config.extra_time_ce - self.config.extra_time,
        )
        if not isinstance(infer_wav, torch.Tensor):
            raise RuntimeError("custom_infer must return a torch.Tensor")
        infer_wav = infer_wav.to(self.device)
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav.unsqueeze(0)).squeeze(0)
        if infer_wav.numel() == 0:
            return torch.zeros(
                self.block_frame, device=self.device, dtype=torch.float32
            )
        return self._apply_sola(infer_wav)

    def flush(self) -> Optional[bytes]:
        # If already flushed, return None to prevent infinite loop
        if self._flushed:
            return None

        flush_output = b""

        # Process any remaining data in the buffer, bypassing VAD check to ensure all data is processed
        infer_wav = self._run_inference(force=True)
        if infer_wav is not None:
            infer_wav = infer_wav.clamp_(-1.0, 1.0)
            flush_output += (infer_wav.cpu().numpy() * 32767.0).astype(np.int16).tobytes()

        # Now add the remaining content from the SOLA buffer (this contains the end of the audio that would otherwise be lost)
        if self.sola_buffer_has_data and self.sola_buffer.numel() > 0:
            sola_audio = self.sola_buffer.clamp_(-1.0, 1.0)
            flush_output += (sola_audio.cpu().numpy() * 32767.0).astype(np.int16).tobytes()

        # Clean up session state to ensure proper shutdown
        self._flushed = True
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        self.vad_cache.clear()

        # Return the complete flushed audio if we have any
        return flush_output if flush_output else None


class RealTimeVCService:
    def __init__(
        self,
        voice_root: str = "examples/reference",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        fp16: Optional[bool] = None,
    ) -> None:
        self.module = _load_realtime_gui_module()
        self.device = _select_device()
        if fp16 is None:
            fp16 = self.device.type in {"cuda", "mps"}
        self.module.device = self.device
        args = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            fp16=fp16,
        )
        self.model_set = self.module.load_models(args)
        self.file_map = FileMap(voice_root, file_extensions=[".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"])
        self.vad_model = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            disable_update=True,
        )

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
    ) -> RealTimeVCSession:
        if voice_id not in self.file_map:
            raise KeyError(f"Unknown voice id: {voice_id}")
        reference_path = self.file_map[voice_id]
        config = RealTimeVCConfig()
        config.apply_overrides(overrides)
        return RealTimeVCSession(
            model_set=self.model_set,
            device=self.device,
            vad_model=self.vad_model,
            reference_audio_path=reference_path,
            source_sample_rate=source_sample_rate,
            config=config,
            module=self.module,
        )


async def stream_realtime_conversion(
    websocket: WebSocket,
    session: RealTimeVCSession,
) -> None:
    await websocket.send_text(
        json.dumps(
            {
                "event": "ready",
                "target_sample_rate": session.output_sample_rate,
            }
        )
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") in {"websocket.disconnect", "websocket.close"}:
                break
            if message.get("bytes") is not None:
                pcm_bytes = message["bytes"]
                converted = await asyncio.to_thread(session.process_pcm, pcm_bytes)
                if converted:
                    await websocket.send_bytes(converted)
            elif message.get("text"):
                payload = json.loads(message["text"])
                if payload.get("event") == "flush":
                    # Loop until all buffered data is flushed
                    flush_count = 0
                    max_flush_iterations = 20  # Safety limit to prevent infinite loops
                    while flush_count < max_flush_iterations:
                        converted = await asyncio.to_thread(session.flush)
                        if converted:
                            await websocket.send_bytes(converted)
                            flush_count += 1
                        else:
                            # No more data, break the loop
                            break
                    if flush_count >= max_flush_iterations:
                        logger.warning(
                            f"Flush loop reached maximum iterations ({max_flush_iterations}), "
                            "forcing exit to prevent infinite loop"
                        )
                    logger.debug(f"Flush completed after {flush_count} iterations")
                    # Send completed event only after all data is sent
                    await websocket.send_text(json.dumps({"event": "completed"}))
                    break
    except WebSocketDisconnect:
        pass
    finally:
        # Connection cleanup - don't send more data as flush loop above should have handled it
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
        except (RuntimeError, Exception):
            # Connection may already be closed by client, ignore
            pass


__all__ = [
    "RealTimeVCService",
    "RealTimeVCSession",
    "RealTimeVCConfig",
    "stream_realtime_conversion",
]

