from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from app.config import CLASS_TRANSLATIONS, FALLBACK_MODEL


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    class_pt: str
    confidence: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, model_path: str, confidence: float) -> None:
        self.requested_model = model_path
        self.confidence = confidence
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.names = dict(self.model.names)

    def _load_model(self, model_path: str) -> YOLO:
        try:
            print(f"Carregando modelo YOLO: {model_path}")
            return YOLO(model_path)
        except Exception as exc:
            print(f"Erro ao carregar o modelo {model_path}: {exc}")
            if model_path == FALLBACK_MODEL:
                raise RuntimeError("Nao foi possivel carregar o modelo YOLO.") from exc
            print(f"Tentando fallback automatico com {FALLBACK_MODEL}...")
            try:
                self.model_path = FALLBACK_MODEL
                return YOLO(FALLBACK_MODEL)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Falha ao carregar o modelo principal e o fallback {FALLBACK_MODEL}."
                ) from fallback_exc

    def validate_classes(self, requested_classes: list[str], show_all: bool) -> list[int] | None:
        if show_all:
            return None

        name_to_id = {name: class_id for class_id, name in self.names.items()}
        valid_ids: list[int] = []

        for class_name in requested_classes:
            if class_name in name_to_id:
                valid_ids.append(name_to_id[class_name])
                continue

            suggestion = self._suggest_class(class_name)
            if suggestion:
                print(
                    f'Atencao: a classe "{class_name}" nao existe no modelo COCO e sera ignorada. '
                    f'Use "{suggestion}" para detectar {CLASS_TRANSLATIONS.get(suggestion, suggestion)}.'
                )
            else:
                print(
                    f'Atencao: a classe "{class_name}" nao existe no modelo COCO e sera ignorada.'
                )

        if not valid_ids:
            print("Nenhuma classe valida foi informada. Detectando todas as classes do modelo.")
            return None

        return valid_ids

    @staticmethod
    def _suggest_class(class_name: str) -> str | None:
        suggestions = {
            "telefone": "cell phone",
            "celular": "cell phone",
            "notebook": "laptop",
            "garrafa": "bottle",
            "pessoa": "person",
            "livro": "book",
            "copo": "cup",
            "cadeira": "chair",
        }
        return suggestions.get(class_name.strip().lower())

    def detect(self, frame: np.ndarray, allowed_class_ids: list[int] | None) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=allowed_class_ids,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.names.get(class_id, str(class_id))
            class_pt = CLASS_TRANSLATIONS.get(class_name, class_name)
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    class_pt=class_pt,
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )
        return detections
