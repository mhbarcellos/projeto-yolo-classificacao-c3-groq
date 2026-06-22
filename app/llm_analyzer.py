from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - fallback para ambiente sem python-dotenv instalado
    load_dotenv = None  # type: ignore[assignment]

from app.config import LOGS_DIR
from app.utils import ensure_directories


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "Você é um especialista em visão computacional, robótica e sistemas inteligentes. "
    "Analise resultados de detecção de objetos gerados por um pipeline YOLO. "
    "Use apenas os dados enviados. Não invente medições, classes ou eventos. "
    "Quando os dados forem insuficientes, diga isso de forma objetiva."
)


@dataclass(frozen=True)
class LLMAnalysisResult:
    success: bool
    status: str
    model: str
    analysis: str
    output_path: Path | None = None
    prompt_path: Path | None = None
    latency_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None
    from_cache: bool = False


def load_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"O arquivo {path} não contém um objeto JSON válido.")
    return data


def build_c2_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Padroniza a saída da C2 para envio à LLM."""
    return {
        "origem": "Projeto C2 - Reconhecimento de Objetos com YOLO",
        "tipo_saida": "resumo_tecnico_de_sessao",
        "contexto": {
            "descricao": "Sistema de visão computacional que detecta objetos em tempo real por webcam, vídeo ou URL.",
            "algoritmo": "YOLO",
            "observacao": "As classes são limitadas ao modelo COCO usado pelo YOLO pré-treinado.",
        },
        "resultado_c2": summary,
    }


def serialize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_user_prompt(payload_json: str) -> str:
    return f"""Analise os dados abaixo produzidos pelo sistema de reconhecimento de objetos da C2.

Objetivo da análise:
1. Resumir o que foi processado.
2. Identificar padrões relevantes nas detecções.
3. Apontar possíveis anomalias ou limitações da execução.
4. Avaliar a confiabilidade geral dos resultados com base nas métricas disponíveis.
5. Recomendar melhorias técnicas para próximas execuções.

Formato obrigatório da resposta:
- Resumo da execução
- Padrões observados
- Anomalias ou limitações
- Recomendações técnicas
- Conclusão objetiva

Dados da C2 em JSON:
{payload_json}
"""


def _usage_value(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(field)
    return int(value) if value is not None else None


def _write_prompt_file(payload_hash: str, model: str, user_prompt: str) -> Path:
    ensure_directories([LOGS_DIR])
    path = LOGS_DIR / f"llm_prompt_{payload_hash[:12]}.txt"
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Modelo: {model}\n\n")
        file.write("SYSTEM PROMPT\n")
        file.write(SYSTEM_PROMPT)
        file.write("\n\nUSER PROMPT\n")
        file.write(user_prompt)
    return path


def _write_analysis_file(
    *,
    payload_hash: str,
    model: str,
    payload: dict[str, Any],
    analysis: str,
    success: bool,
    status: str,
    latency_seconds: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    error: str | None,
    from_cache: bool,
) -> Path:
    ensure_directories([LOGS_DIR])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"llm_analysis_{timestamp}.json"
    content = {
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
        "status": status,
        "modelo": model,
        "temperatura": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "latencia_segundos": latency_seconds,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "payload_hash": payload_hash,
        "from_cache": from_cache,
        "erro": error,
        "analise": analysis,
        "payload_enviado": payload,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    return path


def _cache_path(payload_hash: str, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    return LOGS_DIR / "llm_cache" / f"{safe_model}_{payload_hash}.json"


def _read_cache(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            cached = json.load(file)
        analysis = cached.get("analise")
        return analysis if isinstance(analysis, str) and analysis.strip() else None
    except Exception:
        return None


def _write_cache(path: Path, analysis: str) -> None:
    ensure_directories([path.parent])
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analise": analysis,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def analyze_summary_with_groq(
    summary_path: Path,
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    use_cache: bool = True,
) -> LLMAnalysisResult:
    """Envia o resultado da C2 para a LLM via Groq e salva a análise final.

    A função foi desenhada para não quebrar o sistema principal. Em caso de falta
    de chave, biblioteca ausente, erro de rede ou resposta inválida, ela retorna
    um resultado controlado e registra o problema em arquivo JSON.
    """
    ensure_directories([LOGS_DIR])
    chosen_model = model or DEFAULT_GROQ_MODEL

    try:
        if load_dotenv is not None:
            load_dotenv()

        chosen_model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        summary = load_json_file(summary_path)
        payload = build_c2_payload(summary)
        payload_json = serialize_payload(payload)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        user_prompt = build_user_prompt(payload_json)
        prompt_path = _write_prompt_file(payload_hash, chosen_model, user_prompt)

        cache_path = _cache_path(payload_hash, chosen_model)
        if use_cache:
            cached_analysis = _read_cache(cache_path)
            if cached_analysis is not None:
                output_path = _write_analysis_file(
                    payload_hash=payload_hash,
                    model=chosen_model,
                    payload=payload,
                    analysis=cached_analysis,
                    success=True,
                    status="cache",
                    latency_seconds=0.0,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    error=None,
                    from_cache=True,
                )
                return LLMAnalysisResult(
                    success=True,
                    status="cache",
                    model=chosen_model,
                    analysis=cached_analysis,
                    output_path=output_path,
                    prompt_path=prompt_path,
                    latency_seconds=0.0,
                    from_cache=True,
                )

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            message = (
                "Análise LLM não executada porque a variável de ambiente GROQ_API_KEY não foi encontrada. "
                "Crie um arquivo .env baseado no .env.example ou configure a variável no sistema."
            )
            output_path = _write_analysis_file(
                payload_hash=payload_hash,
                model=chosen_model,
                payload=payload,
                analysis=message,
                success=False,
                status="skipped_missing_api_key",
                latency_seconds=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                error="GROQ_API_KEY ausente",
                from_cache=False,
            )
            return LLMAnalysisResult(
                success=False,
                status="skipped_missing_api_key",
                model=chosen_model,
                analysis=message,
                output_path=output_path,
                prompt_path=prompt_path,
                error="GROQ_API_KEY ausente",
            )

        try:
            from groq import Groq
        except Exception as exc:
            message = "Análise LLM não executada porque a biblioteca groq não está instalada. Rode: pip install -r requirements.txt"
            output_path = _write_analysis_file(
                payload_hash=payload_hash,
                model=chosen_model,
                payload=payload,
                analysis=message,
                success=False,
                status="skipped_missing_dependency",
                latency_seconds=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                error=str(exc),
                from_cache=False,
            )
            return LLMAnalysisResult(
                success=False,
                status="skipped_missing_dependency",
                model=chosen_model,
                analysis=message,
                output_path=output_path,
                prompt_path=prompt_path,
                error=str(exc),
            )

        client = Groq(api_key=api_key, timeout=timeout_seconds)
        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_seconds = round(time.perf_counter() - started_at, 3)
        analysis = response.choices[0].message.content or "A LLM não retornou conteúdo textual."
        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens")

        if use_cache:
            _write_cache(cache_path, analysis)

        output_path = _write_analysis_file(
            payload_hash=payload_hash,
            model=chosen_model,
            payload=payload,
            analysis=analysis,
            success=True,
            status="ok",
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error=None,
            from_cache=False,
        )
        return LLMAnalysisResult(
            success=True,
            status="ok",
            model=chosen_model,
            analysis=analysis,
            output_path=output_path,
            prompt_path=prompt_path,
            latency_seconds=latency_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    except Exception as exc:
        message = f"Falha controlada ao executar análise LLM: {exc}"
        output_path = None
        try:
            fallback_payload = {"summary_path": str(summary_path)}
            output_path = _write_analysis_file(
                payload_hash="erro",
                model=chosen_model,
                payload=fallback_payload,
                analysis=message,
                success=False,
                status="error",
                latency_seconds=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                error=str(exc),
                from_cache=False,
            )
        except Exception:
            pass
        return LLMAnalysisResult(
            success=False,
            status="error",
            model=chosen_model,
            analysis=message,
            output_path=output_path,
            error=str(exc),
        )
