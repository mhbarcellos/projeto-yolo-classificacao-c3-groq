from __future__ import annotations

from pathlib import Path

from app.config import LOGS_DIR
from app.llm_analyzer import analyze_summary_with_groq


def main() -> None:
    summary_path = LOGS_DIR / "session_summary.json"
    if not summary_path.exists():
        print(f"Resumo da C2 não encontrado em: {summary_path}")
        print("Execute primeiro o sistema YOLO para gerar outputs/logs/session_summary.json.")
        return

    result = analyze_summary_with_groq(Path(summary_path))
    print("\n=== Análise da LLM ===\n")
    print(result.analysis)
    print("\n=== Arquivos gerados ===")
    if result.prompt_path:
        print(f"Prompt salvo em: {result.prompt_path}")
    if result.output_path:
        print(f"Análise salva em: {result.output_path}")
    print(f"Status: {result.status}")
    print(f"Modelo: {result.model}")
    if result.latency_seconds is not None:
        print(f"Latência: {result.latency_seconds}s")
    if result.total_tokens is not None:
        print(f"Tokens totais: {result.total_tokens}")


if __name__ == "__main__":
    main()
