from __future__ import annotations

import json
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.vc_service import VCService, stream_conversion

app = FastAPI(title="Seed VC V2 Backend", version="0.1.0")


CONFIG_CASTERS: Dict[str, callable] = {
    "diffusion_steps": int,
    "length_adjust": float,
    "inference_cfg_rate": float,
    "top_p": float,
    "temperature": float,
    "repetition_penalty": float,
    "chunk_seconds": float,
}


def get_service() -> VCService:
    service = getattr(app.state, "vc_service", None)
    if service is None:
        app.state.vc_service = VCService()
        service = app.state.vc_service
    return service


@app.on_event("startup")
async def startup_event() -> None:
    get_service()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/voices")
async def list_voices() -> Dict[str, object]:
    voices = [
        {"id": voice.id, "title": voice.title}
        for voice in get_service().list_voices()
    ]
    return {"voices": voices}


@app.websocket("/ws/convert")
async def websocket_convert(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        voice_id = websocket.query_params.get("voice_id")
        if not voice_id:
            await websocket.send_text(json.dumps({"event": "error", "message": "voice_id is required"}))
            await websocket.close(code=1008)
            return
        sample_rate_param = websocket.query_params.get("sample_rate", "16000")
        try:
            source_sample_rate = int(sample_rate_param)
        except ValueError:
            await websocket.send_text(json.dumps({"event": "error", "message": "sample_rate must be an integer"}))
            await websocket.close(code=1008)
            return
        overrides = {}
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
            session = get_service().create_session(
                voice_id=voice_id,
                source_sample_rate=source_sample_rate,
                overrides=overrides or None,
            )
        except KeyError:
            await websocket.send_text(json.dumps({"event": "error", "message": "Unknown voice id"}))
            await websocket.close(code=1008)
            return
        await stream_conversion(websocket, session)
    except WebSocketDisconnect:
        return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("svc_backend:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
