#!/usr/bin/env python3
"""
Test client for svc_backend WebSocket API.
Reads local audio file, sends it through WebSocket, receives converted audio, and saves to file.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def convert_audio(
    audio_path: str,
    output_path: str,
    backend_url: str = "ws://localhost:8000",
    voice_id: str = "azuma_0",
    source_sample_rate: int = 16000,
    chunk_size: int = None,
    **config_overrides,
) -> None:
    """
    Convert audio file through WebSocket backend.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save converted audio file
        backend_url: WebSocket backend URL (default: ws://localhost:8000)
        voice_id: Target voice ID (default: azuma_0)
        source_sample_rate: Source audio sample rate (default: 16000)
        chunk_size: Chunk size for sending audio data (default: sample_rate // 4)
        **config_overrides: Additional configuration parameters to override
    """
    # Load audio file
    logger.info(f"Loading audio file: {audio_path}")
    audio_data, file_sample_rate = sf.read(audio_path, dtype="float32")
    logger.info(
        f"Audio loaded: sample_rate={file_sample_rate}, "
        f"duration={len(audio_data)/file_sample_rate:.2f}s, "
        f"shape={audio_data.shape}"
    )

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
        logger.info("Converted stereo to mono")

    # Resample if needed
    if file_sample_rate != source_sample_rate:
        import librosa

        logger.info(f"Resampling from {file_sample_rate}Hz to {source_sample_rate}Hz")
        audio_data = librosa.resample(
            audio_data, orig_sr=file_sample_rate, target_sr=source_sample_rate
        )

    # Clip and convert to PCM int16
    audio_data = np.clip(audio_data, -1.0, 1.0)
    pcm_int16 = (audio_data * 32767).astype(np.int16)
    logger.info(f"Converted to PCM: {pcm_int16.size} samples, {len(pcm_int16.tobytes())} bytes")

    # Set default chunk size
    if chunk_size is None:
        chunk_size = max(source_sample_rate // 4, 1)  # 250ms chunks

    # Build WebSocket URL with query parameters
    ws_url = f"{backend_url}/ws/convert?voice_id={voice_id}&sample_rate={source_sample_rate}"
    for key, value in config_overrides.items():
        ws_url += f"&{key}={value}"

    logger.info(f"Connecting to WebSocket: {ws_url}")

    # Connect and process
    async with websockets.connect(ws_url) as websocket:
        logger.info("WebSocket connected, waiting for ready event")

        # Receive ready event
        ready_message = await websocket.recv()
        ready_data = json.loads(ready_message)

        if ready_data.get("event") == "error":
            error_msg = ready_data.get("message", "Unknown error")
            logger.error(f"WebSocket returned error: {error_msg}")
            raise RuntimeError(f"WebSocket error: {error_msg}")

        if ready_data.get("event") != "ready":
            logger.error(f"Unexpected event: {ready_data}")
            raise RuntimeError(f"Expected 'ready' event, got: {ready_data}")

        target_sample_rate = ready_data["target_sample_rate"]
        logger.info(f"Received ready event: target_sample_rate={target_sample_rate}")

        # Send audio data in chunks
        total_chunks = (pcm_int16.size + chunk_size - 1) // chunk_size
        logger.info(f"Sending {total_chunks} chunks of PCM data (chunk_size={chunk_size})")

        for idx, start in enumerate(range(0, pcm_int16.size, chunk_size)):
            chunk = pcm_int16[start : start + chunk_size]
            if chunk.size == 0:
                continue
            await websocket.send(chunk.tobytes())
            if (idx + 1) % 10 == 0:
                logger.debug(f"Sent {idx + 1}/{total_chunks} chunks")

        logger.info("All PCM data sent, sending flush event")
        await websocket.send(json.dumps({"event": "flush"}))

        # Receive converted audio
        outputs = bytearray()
        completed = False
        chunk_count = 0

        logger.info("Receiving converted audio...")
        while not completed:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for response")
                break

            if isinstance(message, bytes):
                outputs.extend(message)
                chunk_count += 1
                if chunk_count % 10 == 0:
                    logger.debug(
                        f"Received {chunk_count} audio chunks, total bytes: {len(outputs)}"
                    )
            elif isinstance(message, str):
                payload = json.loads(message)
                event_type = payload.get("event")

                if event_type == "error":
                    error_msg = payload.get("message", "Unknown error")
                    logger.error(f"WebSocket returned error event: {error_msg}")
                    raise RuntimeError(f"WebSocket error: {error_msg}")
                elif event_type == "completed":
                    logger.info("Received completed event")
                    completed = True
                else:
                    logger.info(f"Received text event: {payload}")

    logger.info(f"Received total audio data: {len(outputs)} bytes")
    if not outputs:
        raise RuntimeError("No audio payload was received from the websocket")

    # Convert received PCM bytes to numpy array
    output_int16 = np.frombuffer(bytes(outputs), dtype=np.int16)
    output_float = output_int16.astype(np.float32) / 32767.0

    # Save output audio file
    logger.info(f"Saving converted audio to: {output_path}")
    sf.write(output_path, output_float, target_sample_rate)
    logger.info(f"Audio file saved successfully: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test client for svc_backend WebSocket API"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input audio file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output audio file path",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="ws://localhost:8105",
        help="WebSocket backend URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--voice-id",
        type=str,
        default="azuma_0",
        help="Target voice ID (default: azuma_0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Source audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for sending audio data (default: sample_rate // 4)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help="Override diffusion_steps config",
    )
    parser.add_argument(
        "--inference-cfg-rate",
        type=float,
        default=None,
        help="Override inference_cfg_rate config",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=float,
        default=None,
        help="Override max_prompt_length config",
    )

    args = parser.parse_args()

    # Build config overrides
    config_overrides = {}
    if args.diffusion_steps is not None:
        config_overrides["diffusion_steps"] = args.diffusion_steps
    if args.inference_cfg_rate is not None:
        config_overrides["inference_cfg_rate"] = args.inference_cfg_rate
    if args.max_prompt_length is not None:
        config_overrides["max_prompt_length"] = args.max_prompt_length

    # Run async conversion
    asyncio.run(
        convert_audio(
            audio_path=args.input,
            output_path=args.output,
            backend_url=args.backend_url,
            voice_id=args.voice_id,
            source_sample_rate=args.sample_rate,
            chunk_size=args.chunk_size,
            **config_overrides,
        )
    )


if __name__ == "__main__":
    main()

