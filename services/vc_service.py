from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from hydra.utils import instantiate
from omegaconf import DictConfig

from services.voice_library import VoiceLibrary, VoiceProfile


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ConversionConfig:
    diffusion_steps: int = 30
    length_adjust: float = 1.0
    inference_cfg_rate: float = 0.7
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.5
    chunk_seconds: float = 2.0


class ConversionSession:
    """Maintains per-connection state for incremental conversion."""

    def __init__(
        self,
        voice: VoiceProfile,
        wrapper,
        device: torch.device,
        dtype: torch.dtype,
        source_sample_rate: int,
        config: ConversionConfig,
    ) -> None:
        self.voice = voice
        self.wrapper = wrapper
        self.device = device
        self.dtype = dtype
        self.source_sample_rate = source_sample_rate
        self.config = config
        self.target_sr = getattr(wrapper, "sr", 22050)
        self.buffer = np.array([], dtype=np.float32)
        self.chunk_samples = max(int(self.target_sr * self.config.chunk_seconds), self.target_sr // 2)

    def _convert_chunk(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, chunk, self.target_sr)
            converted = self.wrapper.convert_voice(
                source_audio_path=tmp.name,
                target_audio_path=self.voice.path,
                diffusion_steps=self.config.diffusion_steps,
                length_adjust=self.config.length_adjust,
                inference_cfg_rate=self.config.inference_cfg_rate,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
                device=self.device,
                dtype=self.dtype,
            )
        converted_wave = converted.squeeze().astype(np.float32)
        converted_wave = np.clip(converted_wave, -1.0, 1.0)
        converted_int16 = (converted_wave * 32767.0).astype(np.int16)
        return converted_int16.tobytes()

    def process_audio(self, audio_bytes: bytes) -> List[bytes]:
        if not audio_bytes:
            return []
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return []
        audio_float = audio_int16.astype(np.float32) / 32768.0
        if self.source_sample_rate != self.target_sr:
            audio_float = librosa.resample(
                audio_float,
                orig_sr=self.source_sample_rate,
                target_sr=self.target_sr,
            )
        if self.buffer.size == 0:
            self.buffer = audio_float
        else:
            self.buffer = np.concatenate([self.buffer, audio_float])

        outputs: List[bytes] = []
        while self.buffer.size >= self.chunk_samples:
            chunk = self.buffer[: self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples :]
            outputs.append(self._convert_chunk(chunk))
        return outputs

    def flush(self) -> List[bytes]:
        if self.buffer.size == 0:
            return []
        chunk = self.buffer
        self.buffer = np.array([], dtype=np.float32)
        return [self._convert_chunk(chunk)]


class VCService:
    """High-level facade exposing voice list and websocket sessions."""

    def __init__(
        self,
        voice_root: str = "examples/reference",
        ar_checkpoint_path: Optional[str] = None,
        cfm_checkpoint_path: Optional[str] = None,
        compile_ar: bool = False,
        chunk_seconds: float = 2.0,
    ) -> None:
        self.voice_library = VoiceLibrary(voice_root)
        self.device = _select_device()
        self.dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
        self.wrapper = self._load_wrapper(ar_checkpoint_path, cfm_checkpoint_path, compile_ar)
        self.config_template = ConversionConfig(chunk_seconds=chunk_seconds)

    def _load_wrapper(
        self,
        ar_checkpoint_path: Optional[str],
        cfm_checkpoint_path: Optional[str],
        compile_ar: bool,
    ):
        cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
        wrapper = instantiate(cfg)
        wrapper.load_checkpoints(ar_checkpoint_path=ar_checkpoint_path, cfm_checkpoint_path=cfm_checkpoint_path)
        wrapper.to(self.device)
        wrapper.eval()
        wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=self.dtype, device=self.device)
        if compile_ar:
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            if hasattr(torch._inductor.config, "fx_graph_cache"):
                torch._inductor.config.fx_graph_cache = True
            wrapper.compile_ar()
        return wrapper

    def list_voices(self) -> List[VoiceProfile]:
        return self.voice_library.list()

    def create_session(
        self,
        voice_id: str,
        source_sample_rate: int,
        overrides: Optional[Dict[str, float]] = None,
    ) -> ConversionSession:
        voice = self.voice_library.get(voice_id)
        config_data: Dict[str, float] = {
            "diffusion_steps": self.config_template.diffusion_steps,
            "length_adjust": self.config_template.length_adjust,
            "inference_cfg_rate": self.config_template.inference_cfg_rate,
            "top_p": self.config_template.top_p,
            "temperature": self.config_template.temperature,
            "repetition_penalty": self.config_template.repetition_penalty,
            "chunk_seconds": self.config_template.chunk_seconds,
        }
        if overrides:
            config_data.update(overrides)
        config = ConversionConfig(**config_data)
        return ConversionSession(
            voice=voice,
            wrapper=self.wrapper,
            device=self.device,
            dtype=self.dtype,
            source_sample_rate=source_sample_rate,
            config=config,
        )


async def stream_conversion(
    websocket: WebSocket,
    session: ConversionSession,
) -> None:
    await websocket.send_text(json.dumps({"event": "ready", "target_sample_rate": session.target_sr}))
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") in {"websocket.disconnect", "websocket.close"}:
                break
            if "bytes" in message and message["bytes"] is not None:
                chunks = await asyncio.to_thread(session.process_audio, message["bytes"])
                for chunk in chunks:
                    await websocket.send_bytes(chunk)
            elif "text" in message and message["text"]:
                payload = json.loads(message["text"])
                if payload.get("event") == "flush":
                    chunks = await asyncio.to_thread(session.flush)
                    for chunk in chunks:
                        await websocket.send_bytes(chunk)
                    await websocket.send_text(json.dumps({"event": "completed"}))
    except WebSocketDisconnect:
        pass
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            remaining = await asyncio.to_thread(session.flush)
            for chunk in remaining:
                await websocket.send_bytes(chunk)
            await websocket.close()


__all__ = [
    "VCService",
    "ConversionSession",
    "ConversionConfig",
    "stream_conversion",
]
