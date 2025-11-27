from __future__ import annotations

import json
import logging
import os
from typing import Dict, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.realtime_vc_service import (
    RealTimeVCService,
    stream_realtime_conversion,
)
from services.refine_onnx import RefineONNXService

app = FastAPI(title="Seed VC V2 Backend", version="0.1.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_CASTERS: Dict[str, callable] = {
    "diffusion_steps": int,
    "inference_cfg_rate": float,
    "max_prompt_length": float,
    "block_time": float,
    "crossfade_time": float,
    "extra_time_ce": float,
    "extra_time": float,
    "extra_time_right": float,
    "pitch_shift": float,
}


def get_service() -> Union[RealTimeVCService, RefineONNXService]:
    service = getattr(app.state, "vc_service", None)
    if service is None:
        use_onnx = os.environ.get("USE_ONNX", "0") == "1"
        if use_onnx:
            logger.info("Initializing RefineONNXService and loading models...")
            # Ensure onnx_models directory exists or is specified correctly
            onnx_dir = os.environ.get("ONNX_DIR", "onnx_models")
            app.state.vc_service = RefineONNXService(onnx_dir=onnx_dir)
        else:
            logger.info("Initializing RealTimeVCService and loading models...")
            app.state.vc_service = RealTimeVCService()
            
        service = app.state.vc_service
        logger.info("Models loaded successfully!")
        logger.info(f"Available voices: {len(service.list_voices())}")
    return service


@app.on_event("startup")
async def startup_event() -> None:
    """
    Preload all models during server startup to avoid delay on first request.
    """
    logger.info("Starting up server, preloading models...")
    try:
        service = get_service()
        # Warm up the model by creating a test session (optional, but ensures everything is ready)
        voices = service.list_voices()
        if voices:
            # Use the first available voice for warmup
            first_voice_id = voices[0]["id"]
            logger.info(f"Warming up model with voice: {first_voice_id}")
            try:
                test_session = service.create_session(
                    voice_id=first_voice_id,
                    source_sample_rate=16000,
                    overrides=None,
                )
                logger.info("Model warmup completed successfully")
            except Exception as e:
                logger.warning(f"Model warmup failed (non-critical): {e}")
        logger.info("Server startup completed, ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}", exc_info=True)
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/voices")
async def list_voices() -> Dict[str, object]:
    return {"voices": get_service().list_voices()}


@app.websocket("/ws/convert")
async def websocket_convert(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        voice_id = websocket.query_params.get("voice_id")
        if not voice_id:
            voice_id = "azuma_0"  # Default to azuma_0.wav if not provided
        sample_rate_param = websocket.query_params.get("sample_rate", "16000")
        try:
            source_sample_rate = int(sample_rate_param)
        except ValueError:
            await websocket.send_text(json.dumps({"event": "error", "message": "sample_rate must be an integer"}))
            await websocket.close(code=1008)
            return
        overrides: Dict[str, float] = {}
        for key, caster in CONFIG_CASTERS.items():
            if key in websocket.query_params:
                try:
                    overrides[key] = caster(websocket.query_params[key])
                except ValueError:
                    await websocket.send_text(
                        json.dumps({"event": "error", "message": f"Invalid value for {key}"})
                    )
                    await websocket.close(code=1008)
                    return
        try:
            service = get_service()
            session = service.create_session(
                voice_id=voice_id,
                source_sample_rate=source_sample_rate,
                overrides=overrides or None,
            )
        except KeyError:
            await websocket.send_text(json.dumps({"event": "error", "message": "Unknown voice id"}))
            await websocket.close(code=1008)
            return
        await stream_realtime_conversion(websocket, session)
    except WebSocketDisconnect:
        return


def run(host: str = "0.0.0.0", port: int = 8105) -> None:
    import uvicorn

    uvicorn.run("svc_backend:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
