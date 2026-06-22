from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from app.config import DASHBOARD_COLORS, PROJECT_NAME
from app.detector import Detection
from app.utils import format_duration


def class_color(class_name: str) -> tuple[int, int, int]:
    seed = sum(ord(char) for char in class_name)
    palette = [
        (56, 189, 248),
        (52, 211, 153),
        (251, 191, 36),
        (248, 113, 113),
        (167, 139, 250),
        (34, 197, 94),
        (244, 114, 182),
        (45, 212, 191),
    ]
    return palette[seed % len(palette)]


def draw_transparent_rect(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, start, end, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        color = class_color(detection.class_name)
        label = f"{detection.class_pt} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        text_w, text_h = text_size
        label_y = max(y1, text_h + 12)
        draw_transparent_rect(
            frame,
            (x1, label_y - text_h - 10),
            (min(x1 + text_w + 12, width - 1), label_y + 5),
            color,
            0.88,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            DASHBOARD_COLORS["dark"],
            2,
            cv2.LINE_AA,
        )


def draw_dashboard(
    frame: np.ndarray,
    *,
    model_name: str,
    fps: float,
    elapsed_seconds: float,
    frame_count: int,
    total_detections: int,
    class_counts: Counter[str],
    confidence_average: float,
    logs_enabled: bool,
    recording: bool,
) -> None:
    height, width = frame.shape[:2]
    panel_width = min(390, max(320, width // 3))
    panel_height = min(height - 24, 430)
    x0, y0 = 12, 12
    x1, y1 = x0 + panel_width, y0 + panel_height

    draw_transparent_rect(frame, (x0, y0), (x1, y1), DASHBOARD_COLORS["panel"], 0.78)
    cv2.rectangle(frame, (x0, y0), (x1, y1), DASHBOARD_COLORS["panel_secondary"], 1)

    title_y = y0 + 30
    cv2.putText(frame, PROJECT_NAME, (x0 + 16, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, DASHBOARD_COLORS["text"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"Modelo: {model_name}", (x0 + 16, title_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.49, DASHBOARD_COLORS["muted"], 1, cv2.LINE_AA)

    lines = [
        f"FPS: {fps:.1f}",
        f"Tempo: {format_duration(elapsed_seconds)}",
        f"Frames: {frame_count}",
        f"Deteccoes: {total_detections}",
        f"Conf. media: {confidence_average:.2f}",
        f"Logs: {'ON' if logs_enabled else 'OFF'}",
        f"Gravacao: {'ON' if recording else 'OFF'}",
    ]

    y = title_y + 66
    for line in lines:
        color = DASHBOARD_COLORS["success"] if line.endswith("ON") else DASHBOARD_COLORS["text"]
        if line.endswith("OFF"):
            color = DASHBOARD_COLORS["danger"]
        cv2.putText(frame, line, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 24

    y += 8
    cv2.putText(frame, "Top classes detectadas", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, DASHBOARD_COLORS["warning"], 1, cv2.LINE_AA)
    y += 24

    top_classes = class_counts.most_common(4)
    if not top_classes:
        cv2.putText(frame, "Aguardando deteccoes...", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, DASHBOARD_COLORS["muted"], 1, cv2.LINE_AA)
        y += 24
    else:
        for label, count in top_classes:
            cv2.putText(frame, f"{label}: {count}", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, DASHBOARD_COLORS["text"], 1, cv2.LINE_AA)
            y += 23

    y = y1 - 78
    cv2.line(frame, (x0 + 16, y - 12), (x1 - 16, y - 12), DASHBOARD_COLORS["panel_secondary"], 1)
    shortcuts = ["S = print", "R = gravar", "L = logs", "Q = sair"]
    for index, shortcut in enumerate(shortcuts):
        col = index % 2
        row = index // 2
        cv2.putText(
            frame,
            shortcut,
            (x0 + 16 + col * 150, y + row * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            DASHBOARD_COLORS["muted"],
            1,
            cv2.LINE_AA,
        )
