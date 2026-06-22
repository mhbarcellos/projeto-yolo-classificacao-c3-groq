from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.utils import parse_source


class VideoStream:
    def __init__(self, source: str) -> None:
        self.source_raw = str(source)
        self.source = parse_source(source)
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            hint = "Teste --source 0 ou --source 1 para webcam."
            if isinstance(self.source, str) and not Path(self.source).exists():
                hint = "O arquivo informado nao foi encontrado."
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}. {hint}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    @property
    def width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280

    @property
    def height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    @property
    def fps(self) -> float:
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        return fps if 1 < fps <= 120 else 30.0

    def release(self) -> None:
        self.capture.release()
