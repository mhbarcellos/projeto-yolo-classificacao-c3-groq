from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import LOGS_DIR, PRINTS_DIR, VIDEOS_DIR
from app.detector import Detection
from app.utils import average, ensure_directories


class SessionLogger:
    def __init__(
        self,
        model_name: str,
        source: str,
        confidence: float,
        configured_classes: list[str],
        show_all: bool,
        logs_enabled: bool,
    ) -> None:
        ensure_directories([LOGS_DIR, PRINTS_DIR, VIDEOS_DIR])
        self.model_name = model_name
        self.source = source
        self.confidence = confidence
        self.configured_classes = configured_classes
        self.show_all = show_all
        self.logs_enabled = logs_enabled
        self.started_at = datetime.now()
        self.ended_at: datetime | None = None

        self.csv_path = LOGS_DIR / "detections.csv"
        self.summary_path = LOGS_DIR / "session_summary.json"
        self.prints_saved: list[str] = []
        self.video_generated = ""
        self.total_detections = 0
        self.class_counts: Counter[str] = Counter()
        self.confidences: list[float] = []
        self._closed = False

        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file, delimiter=";")
        self._writer.writerow(
            ["data_hora", "frame", "classe_original", "classe_pt", "confianca", "x1", "y1", "x2", "y2"]
        )

    def set_logs_enabled(self, enabled: bool) -> None:
        self.logs_enabled = enabled

    def register_detections(self, frame_number: int, detections: list[Detection]) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for detection in detections:
            self.total_detections += 1
            self.class_counts[detection.class_pt] += 1
            self.confidences.append(detection.confidence)

            if not self.logs_enabled:
                continue

            x1, y1, x2, y2 = detection.box
            try:
                self._writer.writerow(
                    [
                        timestamp,
                        frame_number,
                        detection.class_name,
                        detection.class_pt,
                        f"{detection.confidence:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )
            except Exception as exc:
                print(f"Erro ao salvar log CSV: {exc}")

    def save_print(self, frame: np.ndarray) -> Path | None:
        path = PRINTS_DIR / f"print_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            ok = cv2.imwrite(str(path), frame)
            if not ok:
                raise RuntimeError("cv2.imwrite retornou False")
            self.prints_saved.append(str(path))
            print(f"Print salvo em: {path}")
            return path
        except Exception as exc:
            print(f"Erro ao salvar print: {exc}")
            return None

    def register_video(self, path: Path) -> None:
        self.video_generated = str(path)

    def confidence_average(self) -> float:
        return average(self.confidences)

    def finish(self, total_frames: int, average_fps: float) -> Path:
        if self._closed:
            return self.summary_path

        self.ended_at = datetime.now()
        duration = (self.ended_at - self.started_at).total_seconds()
        self._csv_file.flush()
        self._csv_file.close()
        self._closed = True

        summary = {
            "data_hora_inicio": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "data_hora_fim": self.ended_at.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_segundos": round(duration, 2),
            "modelo_utilizado": self.model_name,
            "fonte_video": self.source,
            "confianca_minima": self.confidence,
            "classes_configuradas": self.configured_classes,
            "modo_show_all": self.show_all,
            "total_frames_processados": total_frames,
            "fps_medio": round(average_fps, 2),
            "total_deteccoes": self.total_detections,
            "classes_detectadas": sorted(self.class_counts.keys()),
            "quantidade_por_classe": dict(self.class_counts),
            "confianca_media": round(self.confidence_average(), 4),
            "prints_salvos": self.prints_saved,
            "video_gerado": self.video_generated,
            "logs_ativos_no_encerramento": self.logs_enabled,
        }

        try:
            with open(self.summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Erro ao salvar resumo JSON: {exc}")

        return self.summary_path
