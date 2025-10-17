from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from utils.audio_assets import FileMap


@dataclass(frozen=True)
class VoiceProfile:
    """Static metadata describing an available reference voice."""

    id: str
    title: str
    path: str


class VoiceLibrary:
    """Discovers and exposes reference voices for zero-shot conversion."""

    def __init__(
        self,
        root_dir: str = "examples/reference",
        file_extensions: Optional[Iterable[str]] = None,
    ) -> None:
        extensions = list(file_extensions) if file_extensions is not None else [".wav"]
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Voice directory '{root_dir}' does not exist")
        self._root_dir = root_dir
        self._file_map = FileMap(root_dir, recursive=False, file_extensions=extensions)
        self._profiles: Dict[str, VoiceProfile] = {}
        self._hydrate()

    @staticmethod
    def _slug_to_title(slug: str) -> str:
        tokens = slug.replace("_", " ").replace("-", " ").split()
        return " ".join(token.capitalize() for token in tokens) if tokens else slug

    @staticmethod
    def _resolve_path(rel_path: str) -> str:
        normalized = rel_path[2:] if rel_path.startswith("./") else rel_path
        return os.path.abspath(normalized)

    def _hydrate(self) -> None:
        for voice_id, rel_path in self._file_map.get_all_files().items():
            title = self._slug_to_title(voice_id)
            self._profiles[voice_id] = VoiceProfile(
                id=voice_id,
                title=title,
                path=self._resolve_path(rel_path),
            )

    def list(self) -> List[VoiceProfile]:
        return sorted(self._profiles.values(), key=lambda item: item.title)

    def get(self, voice_id: str) -> VoiceProfile:
        try:
            return self._profiles[voice_id]
        except KeyError as exc:
            raise KeyError(f"Unknown voice id '{voice_id}'") from exc


__all__ = ["VoiceLibrary", "VoiceProfile"]
