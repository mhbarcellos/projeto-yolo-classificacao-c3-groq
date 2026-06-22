from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import customtkinter as ctk
from PIL import Image

from app.config import (
    DEFAULT_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    LOGS_DIR,
    OUTPUTS_DIR,
    PRINTS_DIR,
    PROJECT_NAME,
    VIDEOS_DIR,
)
from app.dashboard import draw_dashboard, draw_detections
from app.detector import ObjectDetector
from app.llm_analyzer import analyze_summary_with_groq
from app.session_logger import SessionLogger
from app.utils import ensure_directories, format_duration, split_classes
from app.video_stream import VideoStream


class DesktopApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ensure_directories([OUTPUTS_DIR, PRINTS_DIR, VIDEOS_DIR, LOGS_DIR])

        self.title(PROJECT_NAME)
        self.geometry("1280x820")
        self.minsize(1120, 720)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.detector: ObjectDetector | None = None
        self.stream: VideoStream | None = None
        self.logger: SessionLogger | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.video_path: Path | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ui_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        self.frame_count = 0
        self.fps_current = 0.0
        self.fps_samples: list[float] = []
        self.started_at = 0.0
        self.last_processed_frame = None
        self.logs_enabled = True
        self.is_running = False

        self.model_var = ctk.StringVar(value=DEFAULT_MODEL)
        self.source_var = ctk.StringVar(value="0")
        self.conf_var = ctk.StringVar(value=str(DEFAULT_CONFIDENCE))
        self.show_all_var = ctk.BooleanVar(value=False)
        self.class_vars = {class_name: ctk.BooleanVar(value=True) for class_name in DEFAULT_CLASSES}

        self._build_layout()
        self._set_idle_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.after(50, self._process_ui_queue)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=270, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        content = ctk.CTkFrame(self, corner_radius=0, fg_color="#111827")
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(content, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))

        bottom = ctk.CTkFrame(content, height=205)
        bottom.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        self.status_text = ctk.CTkTextbox(bottom, height=165, wrap="word")
        self.status_text.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self.status_text.configure(state="disabled")

        self.events_text = ctk.CTkTextbox(bottom, height=165, wrap="word")
        self.events_text.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self.events_text.configure(state="disabled")

        self._build_sidebar(sidebar)

    def _build_sidebar(self, sidebar: ctk.CTkFrame) -> None:
        ctk.CTkLabel(sidebar, text=PROJECT_NAME, font=ctk.CTkFont(size=18, weight="bold"), wraplength=230).pack(padx=18, pady=(22, 14))

        self.start_button = ctk.CTkButton(sidebar, text="Iniciar camera", command=self.start_camera)
        self.start_button.pack(fill="x", padx=18, pady=5)
        self.stop_button = ctk.CTkButton(sidebar, text="Parar", command=self.stop_camera, state="disabled", fg_color="#7f1d1d", hover_color="#991b1b")
        self.stop_button.pack(fill="x", padx=18, pady=5)
        self.print_button = ctk.CTkButton(sidebar, text="Salvar print", command=self.save_print, state="disabled")
        self.print_button.pack(fill="x", padx=18, pady=5)
        self.record_button = ctk.CTkButton(sidebar, text="Iniciar gravacao", command=self.toggle_recording, state="disabled")
        self.record_button.pack(fill="x", padx=18, pady=5)
        self.logs_button = ctk.CTkButton(sidebar, text="Desligar logs", command=self.toggle_logs, state="disabled")
        self.logs_button.pack(fill="x", padx=18, pady=5)
        ctk.CTkButton(sidebar, text="Abrir outputs", command=self.open_outputs).pack(fill="x", padx=18, pady=5)
        ctk.CTkButton(sidebar, text="Encerrar app", command=self.close_app, fg_color="#374151", hover_color="#4b5563").pack(fill="x", padx=18, pady=(5, 16))

        ctk.CTkLabel(sidebar, text="Modelo").pack(anchor="w", padx=18, pady=(8, 2))
        ctk.CTkOptionMenu(sidebar, values=[DEFAULT_MODEL, FALLBACK_MODEL], variable=self.model_var).pack(fill="x", padx=18, pady=3)

        ctk.CTkLabel(sidebar, text="Confianca minima").pack(anchor="w", padx=18, pady=(8, 2))
        ctk.CTkEntry(sidebar, textvariable=self.conf_var).pack(fill="x", padx=18, pady=3)

        ctk.CTkLabel(sidebar, text="Fonte de video").pack(anchor="w", padx=18, pady=(8, 2))
        ctk.CTkEntry(sidebar, textvariable=self.source_var).pack(fill="x", padx=18, pady=3)

        ctk.CTkCheckBox(sidebar, text="Detectar todas as classes", variable=self.show_all_var, command=self._toggle_class_state).pack(anchor="w", padx=18, pady=(12, 4))

        class_box = ctk.CTkScrollableFrame(sidebar, height=230, label_text="Classes")
        class_box.pack(fill="both", expand=True, padx=18, pady=(4, 18))
        self.class_checks: list[ctk.CTkCheckBox] = []
        for class_name, variable in self.class_vars.items():
            checkbox = ctk.CTkCheckBox(class_box, text=class_name, variable=variable)
            checkbox.pack(anchor="w", padx=8, pady=3)
            self.class_checks.append(checkbox)

    def _toggle_class_state(self) -> None:
        state = "disabled" if self.show_all_var.get() else "normal"
        for checkbox in self.class_checks:
            checkbox.configure(state=state)

    def _set_idle_preview(self) -> None:
        preview = Image.new("RGB", (900, 520), "#111827")
        image = ctk.CTkImage(light_image=preview, dark_image=preview, size=(900, 520))
        self.video_label.configure(image=image, text="Clique em Iniciar camera para comecar")
        self.video_label.image = image
        self._update_status_text()

    def start_camera(self) -> None:
        if self.is_running:
            return

        try:
            confidence = float(self.conf_var.get().replace(",", "."))
        except ValueError:
            self._log_event("Confianca invalida. Use um valor como 0.35.")
            return

        selected_classes = [name for name, var in self.class_vars.items() if var.get()]
        if not selected_classes and not self.show_all_var.get():
            self._log_event("Selecione ao menos uma classe ou marque detectar todas.")
            return

        self.stop_event.clear()
        self.frame_count = 0
        self.fps_current = 0.0
        self.fps_samples = []
        self.started_at = time.time()
        self.last_processed_frame = None
        self.logs_enabled = True
        self.video_writer = None
        self.video_path = None
        self.is_running = True
        self._set_running_state(True)
        self._log_event("Iniciando camera e carregando modelo...")

        self.worker_thread = threading.Thread(
            target=self._video_loop,
            args=(self.model_var.get(), self.source_var.get(), confidence, selected_classes, self.show_all_var.get()),
            daemon=True,
        )
        self.worker_thread.start()

    def _video_loop(
        self,
        model_name: str,
        source: str,
        confidence: float,
        selected_classes: list[str],
        show_all: bool,
    ) -> None:
        previous_frame_at = time.time()
        try:
            self.detector = ObjectDetector(model_name, confidence)
            allowed_ids = self.detector.validate_classes(selected_classes, show_all)
            self.stream = VideoStream(source)
            self.logger = SessionLogger(
                model_name=self.detector.model_path,
                source=source,
                confidence=confidence,
                configured_classes=[] if show_all else selected_classes,
                show_all=show_all,
                logs_enabled=self.logs_enabled,
            )
            self._queue_event("Modelo carregado com sucesso.")
            self._queue_event("Camera iniciada.")

            while not self.stop_event.is_set():
                ok, frame = self.stream.read()
                if not ok or frame is None:
                    self._queue_event("Fim da fonte de video ou falha na leitura.")
                    break

                self.frame_count += 1
                detections = self.detector.detect(frame, allowed_ids)
                self.logger.register_detections(self.frame_count, detections)

                now = time.time()
                delta = now - previous_frame_at
                previous_frame_at = now
                if delta > 0:
                    instant_fps = 1.0 / delta
                    self.fps_current = instant_fps if self.fps_current == 0 else (0.85 * self.fps_current + 0.15 * instant_fps)
                    self.fps_samples.append(self.fps_current)

                draw_detections(frame, detections)
                draw_dashboard(
                    frame,
                    model_name=self.detector.model_path,
                    fps=self.fps_current,
                    elapsed_seconds=now - self.started_at,
                    frame_count=self.frame_count,
                    total_detections=self.logger.total_detections,
                    class_counts=self.logger.class_counts,
                    confidence_average=self.logger.confidence_average(),
                    logs_enabled=self.logger.logs_enabled,
                    recording=self.video_writer is not None,
                )

                if self.video_writer is not None:
                    try:
                        self.video_writer.write(frame)
                    except Exception as exc:
                        self._queue_event(f"Erro ao gravar video: {exc}")

                self.last_processed_frame = frame.copy()
                self.ui_queue.put({"type": "frame", "frame": frame})
                self.ui_queue.put({"type": "status"})

        except Exception as exc:
            self._queue_event(f"Erro: {exc}")
        finally:
            self._release_resources()
            self.ui_queue.put({"type": "stopped"})

    def stop_camera(self) -> None:
        if not self.is_running:
            return
        self._log_event("Parando captura...")
        self.stop_event.set()

    def save_print(self) -> None:
        if self.logger is None or self.last_processed_frame is None:
            self._log_event("Nenhum frame processado disponivel para print.")
            return
        path = self.logger.save_print(self.last_processed_frame)
        if path is not None:
            self._log_event(f"Print salvo em {path}")

    def toggle_recording(self) -> None:
        if self.stream is None or self.logger is None:
            self._log_event("Inicie a camera antes de gravar.")
            return

        if self.video_writer is None:
            path = VIDEOS_DIR / f"video_processado_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, self.stream.fps, (self.stream.width, self.stream.height))
            if not writer.isOpened():
                self._log_event(f"Erro ao iniciar gravacao em {path}")
                return
            self.video_writer = writer
            self.video_path = path
            self.logger.register_video(path)
            self.record_button.configure(text="Parar gravacao")
            self._log_event(f"Gravacao iniciada: {path}")
        else:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.record_button.configure(text="Iniciar gravacao")
            self._log_event(f"Gravacao finalizada: {self.video_path}")

    def toggle_logs(self) -> None:
        if self.logger is None:
            self.logs_enabled = not self.logs_enabled
        else:
            self.logger.set_logs_enabled(not self.logger.logs_enabled)
            self.logs_enabled = self.logger.logs_enabled
        self.logs_button.configure(text="Desligar logs" if self.logs_enabled else "Ligar logs")
        self._log_event(f"Logs {'ON' if self.logs_enabled else 'OFF'}")
        self._update_status_text()

    def open_outputs(self) -> None:
        try:
            ensure_directories([OUTPUTS_DIR])
            if hasattr(os, "startfile"):
                os.startfile(OUTPUTS_DIR)  # type: ignore[attr-defined]
            else:
                self._log_event(f"Pasta outputs: {OUTPUTS_DIR}")
            self._log_event("Pasta outputs aberta.")
        except Exception as exc:
            self._log_event(f"Erro ao abrir outputs: {exc}")

    def close_app(self) -> None:
        try:
            self.stop_event.set()
            if self.worker_thread is not None and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3)
            self._release_resources()
        except Exception as exc:
            self._log_event(f"Erro ao finalizar app: {exc}")
        finally:
            self.destroy()

    def _release_resources(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.stream is not None:
            self.stream.release()
            self.stream = None
        if self.logger is not None:
            average_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0
            summary_path = self.logger.finish(self.frame_count, average_fps)
            self._queue_event(f"Resumo bruto da C2 salvo em {summary_path}")
            self._queue_event("Enviando resumo da C2 para analise generativa via Groq...")
            llm_result = analyze_summary_with_groq(Path(summary_path))
            if llm_result.output_path is not None:
                self._queue_event(f"Analise LLM salva em {llm_result.output_path}")
            if llm_result.prompt_path is not None:
                self._queue_event(f"Prompt enviado salvo em {llm_result.prompt_path}")
            self._queue_event(f"Status LLM: {llm_result.status}")
            preview = llm_result.analysis.strip()
            if len(preview) > 1600:
                preview = preview[:1600] + "..."
            self._queue_event("Analise interpretativa da LLM:\n" + preview)
            self.logger = None

    def _process_ui_queue(self) -> None:
        try:
            while True:
                message = self.ui_queue.get_nowait()
                message_type = message.get("type")
                if message_type == "frame":
                    self._show_frame(message["frame"])
                elif message_type == "event":
                    self._log_event(message["text"])
                elif message_type == "status":
                    self._update_status_text()
                elif message_type == "stopped":
                    self.is_running = False
                    self._set_running_state(False)
                    self._update_status_text()
                    self._log_event("Captura finalizada.")
        except queue.Empty:
            pass
        self.after(50, self._process_ui_queue)

    def _show_frame(self, frame: Any) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        label_width = max(self.video_label.winfo_width(), 640)
        label_height = max(self.video_label.winfo_height(), 360)
        image.thumbnail((label_width, label_height))
        tk_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.video_label.configure(image=tk_image, text="")
        self.video_label.image = tk_image

    def _update_status_text(self) -> None:
        elapsed = time.time() - self.started_at if self.started_at else 0
        logger = self.logger
        total_detections = logger.total_detections if logger else 0
        confidence_average = logger.confidence_average() if logger else 0.0
        class_counts = logger.class_counts if logger else {}
        model = self.detector.model_path if self.detector else self.model_var.get()
        logs = logger.logs_enabled if logger else self.logs_enabled
        top_items = class_counts.most_common(5) if hasattr(class_counts, "most_common") else list(class_counts.items())[:5]
        top_classes = "\n".join(f"- {label}: {count}" for label, count in top_items) or "- aguardando deteccoes"

        text = (
            f"Modelo utilizado: {model}\n"
            f"Fonte de video: {self.source_var.get()}\n"
            f"FPS atual: {self.fps_current:.1f}\n"
            f"Tempo de sessao: {format_duration(elapsed)}\n"
            f"Frames processados: {self.frame_count}\n"
            f"Total de deteccoes: {total_detections}\n"
            f"Confianca media: {confidence_average:.2f}\n"
            f"Logs: {'ON' if logs else 'OFF'}\n"
            f"Gravacao: {'ON' if self.video_writer is not None else 'OFF'}\n\n"
            f"Top classes detectadas:\n{top_classes}"
        )
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", text)
        self.status_text.configure(state="disabled")

    def _log_event(self, text: str) -> None:
        print(text)
        timestamp = time.strftime("%H:%M:%S")
        self.events_text.configure(state="normal")
        self.events_text.insert("end", f"[{timestamp}] {text}\n")
        self.events_text.see("end")
        self.events_text.configure(state="disabled")

    def _queue_event(self, text: str) -> None:
        self.ui_queue.put({"type": "event", "text": text})

    def _set_running_state(self, running: bool) -> None:
        normal = "normal"
        disabled = "disabled"
        self.start_button.configure(state=disabled if running else normal)
        self.stop_button.configure(state=normal if running else disabled)
        self.print_button.configure(state=normal if running else disabled)
        self.record_button.configure(state=normal if running else disabled, text="Iniciar gravacao")
        self.logs_button.configure(state=normal if running else disabled, text="Desligar logs" if self.logs_enabled else "Ligar logs")


def run() -> None:
    app = DesktopApp()
    app.mainloop()
