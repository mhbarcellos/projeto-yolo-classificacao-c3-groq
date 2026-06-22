from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.config import (
    DEFAULT_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    DEFAULT_SOURCE,
    LOGS_DIR,
    PRINTS_DIR,
    VIDEOS_DIR,
    WINDOW_NAME,
)
from app.dashboard import draw_dashboard, draw_detections
from app.detector import ObjectDetector
from app.session_logger import SessionLogger
from app.llm_analyzer import analyze_summary_with_groq
from app.utils import ensure_directories, split_classes
from app.video_stream import VideoStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconhecimento de objetos em tempo real com YOLO e OpenCV."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Fonte de video: webcam, arquivo ou URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo YOLO. Exemplo: yolo11n.pt ou yolov8n.pt.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confianca minima das deteccoes.")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Classes COCO em ingles, separadas por virgula.",
    )
    parser.add_argument("--show-all", action="store_true", help="Detecta todas as classes do modelo.")
    parser.add_argument("--save-video", action="store_true", help="Inicia a aplicacao gravando o video processado.")
    parser.add_argument("--no-logs", action="store_true", help="Inicia a aplicacao com os logs CSV desligados.")
    parser.add_argument("--no-llm", action="store_true", help="Nao executa a analise generativa com Groq ao encerrar.")
    parser.add_argument("--llm-model", default=None, help="Modelo Groq para analise final. Exemplo: llama-3.3-70b-versatile.")
    return parser.parse_args()


def create_video_writer(width: int, height: int, fps: float) -> tuple[cv2.VideoWriter | None, Path | None]:
    video_path = VIDEOS_DIR / f"video_processado_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Erro ao iniciar gravacao de video em: {video_path}")
        return None, None
    print(f"Gravacao iniciada: {video_path}")
    return writer, video_path


def stop_video_writer(writer: cv2.VideoWriter | None, video_path: Path | None) -> None:
    if writer is not None:
        writer.release()
        print(f"Gravacao finalizada: {video_path}")


def main() -> None:
    args = parse_args()
    ensure_directories([PRINTS_DIR, VIDEOS_DIR, LOGS_DIR])

    requested_classes = split_classes(args.classes)
    if args.show_all:
        requested_classes = []

    stream: VideoStream | None = None
    video_writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    logger: SessionLogger | None = None

    frame_count = 0
    fps_current = 0.0
    fps_samples: list[float] = []
    started_loop_at = time.time()
    previous_frame_at = time.time()

    try:
        detector = ObjectDetector(args.model, args.conf)
        allowed_class_ids = detector.validate_classes(requested_classes, args.show_all)
        stream = VideoStream(args.source)

        logger = SessionLogger(
            model_name=detector.model_path,
            source=str(args.source),
            confidence=args.conf,
            configured_classes=requested_classes if not args.show_all else [],
            show_all=args.show_all,
            logs_enabled=not args.no_logs,
        )

        if args.save_video:
            video_writer, video_path = create_video_writer(stream.width, stream.height, stream.fps)
            if video_path is not None:
                logger.register_video(video_path)

        print("\nSistema iniciado com sucesso.")
        print("Atalhos: S = print | R = iniciar/parar gravacao | L = ligar/desligar logs | Q = sair")
        print(f"Modelo em uso: {detector.model_path}")
        print(f"Fonte de video: {args.source}")
        print(f"Logs CSV: {'ON' if logger.logs_enabled else 'OFF'}")

        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                print("Fim da fonte de video ou falha na leitura do frame.")
                break

            frame_count += 1
            detections = detector.detect(frame, allowed_class_ids)
            logger.register_detections(frame_count, detections)

            now = time.time()
            delta = now - previous_frame_at
            previous_frame_at = now
            if delta > 0:
                instant_fps = 1.0 / delta
                fps_current = instant_fps if fps_current == 0 else (0.85 * fps_current + 0.15 * instant_fps)
                fps_samples.append(fps_current)

            draw_detections(frame, detections)
            draw_dashboard(
                frame,
                model_name=detector.model_path,
                fps=fps_current,
                elapsed_seconds=now - started_loop_at,
                frame_count=frame_count,
                total_detections=logger.total_detections,
                class_counts=logger.class_counts,
                confidence_average=logger.confidence_average(),
                logs_enabled=logger.logs_enabled,
                recording=video_writer is not None,
            )

            if video_writer is not None:
                video_writer.write(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("Encerrando aplicacao...")
                break

            if key in (ord("s"), ord("S")):
                logger.save_print(frame)

            if key in (ord("l"), ord("L")):
                logger.set_logs_enabled(not logger.logs_enabled)
                print(f"Logs CSV: {'ON' if logger.logs_enabled else 'OFF'}")

            if key in (ord("r"), ord("R")):
                if video_writer is None:
                    video_writer, video_path = create_video_writer(stream.width, stream.height, stream.fps)
                    if video_path is not None:
                        logger.register_video(video_path)
                else:
                    stop_video_writer(video_writer, video_path)
                    video_writer = None

    except Exception as exc:
        print(f"Erro: {exc}")
    finally:
        if video_writer is not None:
            stop_video_writer(video_writer, video_path)
        if stream is not None:
            stream.release()
        cv2.destroyAllWindows()

        if logger is not None:
            average_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
            summary_path = logger.finish(frame_count, average_fps)
            print(f"Resumo tecnico salvo em: {summary_path}")
            print(f"CSV de deteccoes salvo em: {logger.csv_path}")
            print(f"Prints: {PRINTS_DIR}")
            print(f"Videos: {VIDEOS_DIR}")

            if not args.no_llm:
                print("\nEnviando resultado da C2 para analise generativa via Groq...")
                llm_result = analyze_summary_with_groq(Path(summary_path), model=args.llm_model)
                print("\n=== Analise interpretativa da LLM ===\n")
                print(llm_result.analysis)
                if llm_result.prompt_path is not None:
                    print(f"Prompt salvo em: {llm_result.prompt_path}")
                if llm_result.output_path is not None:
                    print(f"Analise LLM salva em: {llm_result.output_path}")
                print(f"Status da chamada LLM: {llm_result.status}")


if __name__ == "__main__":
    main()
