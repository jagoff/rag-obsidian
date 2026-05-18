"""Routes for browsing and editing prompt files."""
from __future__ import annotations

import ast
import fnmatch
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_ROOT = _REPO_ROOT / "rag" / "prompts"
_MAX_PROMPT_CHARS = 200_000

_IMPORTANCE_LABELS = {
    "high": "clave",
    "medium": "media",
    "low": "baja",
}


class PromptWriteRequest(BaseModel):
    path: str
    content: str = Field(..., min_length=1, max_length=_MAX_PROMPT_CHARS)


_INLINE_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "web.chat_config._WEB_SYSTEM_PROMPT_V2",
        "path": "web/chat_config.py",
        "symbol": "_WEB_SYSTEM_PROMPT_V2",
        "importance": "high",
        "purpose": "System prompt principal del web chat: regula uso del contexto, tono, links, citas inline y confirmaciones de tools.",
        "effective": "Es la política central del chat web. Decide cómo usar el contexto recuperado, cuándo puede decir que no encontró datos, cómo citar links, cómo tratar datos financieros y cómo evitar confirmar acciones que no ejecutó.",
        "impact": "Cambiarlo afecta casi todas las respuestas del chat web y puede modificar precisión, tono, grounding y seguridad operativa.",
    },
    {
        "id": "web.chat_config._WEB_WORK_SYSTEM_PROMPT",
        "path": "web/chat_config.py",
        "symbol": "_WEB_WORK_SYSTEM_PROMPT",
        "importance": "medium",
        "purpose": "Modo laboral/AWS/FinOps: permite conocimiento técnico general aunque el contexto del vault sea pobre.",
        "effective": "Se agrega como segundo system prompt cuando el chat entra en modo trabajo. Afloja la restricción de responder sólo desde el vault para permitir razonamiento técnico general.",
        "impact": "Afecta respuestas de trabajo/AWS/FinOps; mal calibrado puede inventar o, al revés, negarse demasiado.",
    },
    {
        "id": "web.server._build_tasks_system_rules",
        "path": "web/server.py",
        "symbol": "_build_tasks_system_rules",
        "importance": "high",
        "purpose": "Prompt dinámico de agenda/tasks por ventana temporal, armado con evidencia de Calendar, Reminders, mails, WhatsApp y vault loops.",
        "effective": "Construye instrucciones para responder sobre agenda y pendientes usando una ventana temporal concreta. Ordena cómo combinar Calendar, Reminders, WhatsApp, mails y loops del vault.",
        "impact": "Afecta directamente qué tareas aparecen como urgentes o relevantes en superficies de planificación.",
    },
    {
        "id": "rag.__init__._AGENT_SYSTEM",
        "path": "rag/__init__.py",
        "symbol": "_AGENT_SYSTEM",
        "importance": "high",
        "purpose": "Prompt del agente de vault con tools de búsqueda, lectura y propuestas de escritura confirmables.",
        "effective": "Define el comportamiento del agente que puede buscar, leer y preparar cambios en el vault. Controla límites de escritura, confirmaciones y estilo de uso de tools.",
        "impact": "Es crítico para seguridad de edición del vault y para que el agente no confirme cambios no realizados.",
    },
    {
        "id": "rag.contextual_retrieval._build_prompt",
        "path": "rag/contextual_retrieval.py",
        "symbol": "_build_prompt",
        "importance": "high",
        "purpose": "Genera una oración de contexto para ubicar un chunk dentro del documento padre durante indexación contextual.",
        "effective": "Durante indexación, le pide al LLM una frase que explique dónde encaja un chunk dentro de su documento. Esa frase mejora retrieval porque agrega contexto semántico al vector.",
        "impact": "Afecta la calidad de búsqueda de todo lo indexado con contexto enriquecido.",
    },
    {
        "id": "rag.query_decompose._llm_decompose",
        "path": "rag/query_decompose.py",
        "symbol": "_llm_decompose",
        "importance": "medium",
        "purpose": "Detecta queries multi-aspecto y devuelve sub-queries autónomas en JSON.",
        "effective": "Divide preguntas compuestas en subconsultas más simples para recuperar evidencia por aspecto y no perder partes de la pregunta.",
        "impact": "Mejora preguntas complejas, pero no gobierna el comportamiento base del sistema.",
    },
    {
        "id": "rag._anaphora._cached_anaphora_resolution",
        "path": "rag/_anaphora.py",
        "symbol": "_cached_anaphora_resolution",
        "importance": "medium",
        "purpose": "Reescribe queries con pronombres, conectores o elipsis usando historial conversacional.",
        "effective": "Convierte preguntas como 'y ella?' o 'profundizá eso' en una query explícita usando el historial reciente.",
        "impact": "Afecta continuidad conversacional y retrieval en follow-ups.",
    },
    {
        "id": "rag.llm_judge._build_prompt",
        "path": "rag/llm_judge.py",
        "symbol": "_build_prompt",
        "importance": "high",
        "purpose": "Juez LLM de candidatos recuperados: asigna scores 0-10 para blending cuando el ranking es incierto.",
        "effective": "Le pide al LLM puntuar candidatos recuperados contra la query. Sirve como señal adicional cuando BM25/vector/reranker no alcanzan para ordenar bien.",
        "impact": "Afecta qué fuentes llegan al modelo en búsquedas ambiguas.",
    },
    {
        "id": "rag.postprocess._NLI_LLM_SYSTEM",
        "path": "rag/postprocess.py",
        "symbol": "_NLI_LLM_SYSTEM",
        "importance": "medium",
        "purpose": "Verifica afirmaciones contra evidencias como entails, neutral o contradicts.",
        "effective": "Clasifica si una afirmación está soportada, contradicha o no determinada por la evidencia. Funciona como chequeo posterior de grounding.",
        "impact": "Afecta validación de respuestas, no la generación primaria.",
    },
    {
        "id": "rag.ocr.vlm_prompts",
        "path": "rag/ocr.py",
        "symbol": "_VLM_*_PROMPT",
        "importance": "medium",
        "purpose": "Prompts VLM para captions generales, recibos, capturas de pantalla y charts.",
        "effective": "Controla cómo se convierten imágenes en texto útil para indexar: captions, tickets, screenshots y gráficos.",
        "impact": "Afecta la calidad de notas creadas desde imágenes y su posterior recuperación.",
    },
    {
        "id": "rag.ocr_cita_detector.cita_prompts",
        "path": "rag/ocr_cita_detector.py",
        "symbol": "_CITA_PROMPT_*",
        "importance": "low",
        "purpose": "Router de OCR reenviado por WhatsApp: evento de calendario, recordatorio o nota en Inbox.",
        "effective": "Clasifica OCR recibido por WhatsApp para decidir si parece cita, reminder o nota común.",
        "impact": "Tiene alcance acotado a capturas/OCR reenviados por WhatsApp.",
    },
    {
        "id": "rag.proactive_draft._build_prompt",
        "path": "rag/proactive_draft.py",
        "symbol": "_build_prompt",
        "importance": "low",
        "purpose": "Redacta un mensaje WhatsApp proactivo sobre una promesa pendiente, mimetizando el estilo del usuario.",
        "effective": "Genera borradores proactivos para cerrar promesas o pendientes detectados en chats.",
        "impact": "Afecta una superficie lateral de drafts, no la respuesta principal del RAG.",
    },
    {
        "id": "rag.integrations.whatsapp.dossier._DOSSIER_PROMPT",
        "path": "rag/integrations/whatsapp/dossier.py",
        "symbol": "_DOSSIER_PROMPT",
        "importance": "medium",
        "purpose": "Dossier compacto de un contacto para que otro LLM tenga contexto al responder o draftar.",
        "effective": "Resume lo relevante de una relación/contacto para inyectarlo como contexto en drafts o respuestas de WhatsApp.",
        "impact": "Afecta calidad y tacto social de respuestas en WhatsApp.",
    },
    {
        "id": "rag.integrations.whatsapp.tasks_extract",
        "path": "rag/integrations/whatsapp/tasks_extract.py",
        "symbol": "prompt",
        "line": 120,
        "importance": "medium",
        "purpose": "Extrae tasks, questions, commitments y promises desde conversaciones de WhatsApp.",
        "effective": "Convierte conversaciones en estructura accionable: tareas, preguntas pendientes, compromisos y promesas.",
        "impact": "Afecta detección de pendientes sociales y loops.",
    },
    {
        "id": "rag.mirror._INSIGHTS_PROMPT",
        "path": "rag/mirror.py",
        "symbol": "_INSIGHTS_PROMPT",
        "importance": "low",
        "purpose": "Genera insights JSON del Personal Mirror con grounding estricto sobre el snapshot.",
        "effective": "Produce insights estructurados del Personal Mirror a partir de un snapshot de señales.",
        "impact": "Impacta una vista analítica específica; no cambia retrieval ni chat principal.",
    },
    {
        "id": "rag.mlx_reranker._SYSTEM",
        "path": "rag/mlx_reranker.py",
        "symbol": "_SYSTEM / _build_prompt",
        "importance": "high",
        "purpose": "Prompt yes/no del reranker MLX para estimar relevancia documento-query.",
        "effective": "Pregunta si un documento es relevante para una query y usa esa señal para reordenar resultados.",
        "impact": "Afecta directamente qué evidencia se considera más relevante.",
    },
]

_PROMPT_INFO_BY_NAME: dict[str, dict[str, str]] = {
    "auto_fix": {
        "importance": "medium",
        "effective": "Guía al agente de reparación automática cuando recibe un error de logs. Define cómo investigar, qué comandos puede usar, cuándo reiniciar daemons y cuándo declarar que hace falta intervención humana.",
        "impact": "Afecta resolución automática de incidentes operativos; no cambia las respuestas normales del chat.",
    },
    "diagnose_error": {
        "importance": "medium",
        "effective": "Convierte un error de UI/log en diagnóstico corto y posibles acciones. Es más explicativo que auto_fix: ayuda a entender el problema antes de ejecutar cambios.",
        "impact": "Afecta la calidad del diagnóstico de errores en la web.",
    },
    "followups": {
        "importance": "medium",
        "effective": "Genera preguntas sugeridas después de una respuesta para continuar explorando el tema con contexto.",
        "impact": "Afecta ergonomía del chat, no el grounding principal.",
    },
    "prep": {
        "importance": "medium",
        "effective": "Arma un brief de preparación sobre un tema/persona/proyecto usando notas recuperadas y señales recientes.",
        "impact": "Afecta comandos de preparación y briefs, no el chat general.",
    },
    "lookup": {
        "importance": "medium",
        "effective": "Responde consultas de conteo, listado, recientes o agenda de forma breve y directa.",
        "impact": "Afecta queries tipo lookup donde importa devolver dato puntual sin síntesis larga.",
    },
    "serve_meta": {
        "importance": "high",
        "effective": "Gobierna el endpoint `/chat` de `rag serve` cuando el sistema necesita responder sobre sí mismo, herramientas, rutas o capacidades.",
        "impact": "Afecta respuestas meta del servicio web y cómo explica sus propias operaciones.",
    },
    "strict": {
        "importance": "high",
        "effective": "Es la variante estricta de consulta: prohíbe marcadores externos y fuerza respuestas muy ancladas al contexto.",
        "impact": "Afecta `rag query` y cualquier superficie que use modo estricto.",
    },
    "system_rules": {
        "importance": "high",
        "effective": "Define las reglas base de respuesta del RAG: uso del contexto, tono, tratamiento del usuario, formato y límites de invención.",
        "impact": "Es uno de los prompts más sensibles: tocarlo cambia el comportamiento general del asistente.",
    },
    "web": {
        "importance": "high",
        "effective": "Adapta las reglas base al chat web, con más espacio para conversación, storytelling y continuidad de hilo.",
        "impact": "Afecta la experiencia diaria del chat web.",
    },
    "chat": {
        "importance": "high",
        "effective": "Variante comprimida para superficies conversacionales tipo WhatsApp/serve, optimizada para respuestas cortas.",
        "impact": "Afecta respuestas donde el canal necesita brevedad y baja latencia.",
    },
    "synthesis": {
        "importance": "high",
        "effective": "Indica cómo sintetizar varias fuentes: cuándo combinar, cuándo negarse por falta de evidencia y cómo presentar conclusiones.",
        "impact": "Afecta respuestas de síntesis multi-fuente y la robustez ante evidencia insuficiente.",
    },
    "comparison": {
        "importance": "high",
        "effective": "Indica cómo comparar fuentes o entidades y cuándo rechazar comparaciones con menos de dos fuentes útiles.",
        "impact": "Afecta respuestas comparativas y evita comparaciones fabricadas.",
    },
    "chunk_as_data": {
        "importance": "high",
        "effective": "Recalca que los chunks recuperados son datos, no instrucciones. Es una defensa contra prompt injection dentro de notas.",
        "impact": "Crítico para seguridad: evita que una nota pueda reprogramar al asistente.",
    },
    "language_es_AR": {
        "importance": "high",
        "effective": "Fuerza español rioplatense argentino y reduce fugas a otros idiomas.",
        "impact": "Afecta todo prompt que lo incluye; tocarlo cambia idioma y tono global.",
    },
    "name_preservation": {
        "importance": "medium",
        "effective": "Preserva nombres propios, alias y entidades tal como aparecen en el contexto.",
        "impact": "Reduce errores de identificación, especialmente con contactos y personas con nombres parecidos.",
    },
}


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_prompt_path(rel_path: str) -> Path:
    rel = (rel_path or "").strip().lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path vacío")
    path = (_REPO_ROOT / rel).resolve()
    try:
        path.relative_to(_PROMPTS_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path fuera de rag/prompts") from exc
    if path.suffix != ".md":
        raise HTTPException(status_code=400, detail="solo se editan prompts .md")
    return path


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
    return value.strip("'\"")


def _dedent_block(lines: list[str]) -> str:
    if not lines:
        return ""
    min_indent: int | None = None
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        min_indent = indent if min_indent is None else min(min_indent, indent)
    if min_indent is None:
        min_indent = 0
    return "\n".join(line[min_indent:] if len(line) >= min_indent else line for line in lines).strip()


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end < 0:
        return {}, raw
    frontmatter = raw[4:end]
    body = raw[end + 5 :]
    meta: dict[str, Any] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush_block() -> None:
        nonlocal current_key, current_lines
        if current_key is not None:
            meta[current_key] = _dedent_block(current_lines)
            current_key = None
            current_lines = []

    for line in frontmatter.splitlines():
        if current_key is not None and (line.startswith(" ") or line.startswith("\t") or not line.strip()):
            current_lines.append(line)
            continue
        flush_block()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            current_key = key
            current_lines = []
        else:
            meta[key] = _parse_scalar(value)
    flush_block()
    return meta, body


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" -\t")
        if stripped:
            return stripped
    return ""


def _purpose(meta: dict[str, Any], body: str) -> str:
    notes = str(meta.get("notes") or "").strip()
    if notes:
        compact = _compact_notes(meta, body, max_chars=260)
        if compact:
            return compact
    body_line = _first_meaningful_line(body)
    if body_line:
        return body_line[:260]
    return "Sin descripción en frontmatter."


def _compact_notes(meta: dict[str, Any], body: str, max_chars: int = 520) -> str:
    text = str(meta.get("notes") or "").strip()
    if not text:
        text = _first_meaningful_line(body)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _prompt_info(name: str, status: str, meta: dict[str, Any], body: str) -> dict[str, str]:
    info = dict(_PROMPT_INFO_BY_NAME.get(name, {}))
    if status in {"legacy", "deprecated"}:
        importance = "low"
        reason = "Versión histórica: queda visible para auditoría o rollback, pero no es la versión activa."
        impact = reason
    else:
        importance = str(meta.get("importance") or info.get("importance") or "medium")
        reason = info.get("impact") or "Afecta una superficie activa del sistema."
        impact = str(meta.get("impact") or info.get("impact") or reason).strip()
    if importance not in _IMPORTANCE_LABELS:
        importance = "medium"
    effective = str(meta.get("effective") or info.get("effective") or "").strip()
    if not effective:
        notes = _compact_notes(meta, body)
        effective = notes or "No tiene una descripción efectiva específica; revisá el cuerpo del prompt para ver sus instrucciones."
    return {
        "importance": importance,
        "importance_label": _IMPORTANCE_LABELS[importance],
        "importance_reason": reason,
        "effective": effective,
        "impact": impact,
    }


def _name_version_from_path(path: Path, meta: dict[str, Any]) -> tuple[str, str]:
    name = str(meta.get("name") or "").strip()
    version = str(meta.get("version") or "").strip()
    stem = path.stem
    if "." in stem:
        stem_name, stem_version = stem.rsplit(".", 1)
        name = name or stem_name
        version = version or stem_version
    else:
        name = name or stem
    return name, version


def _latest_versions(files: list[Path]) -> dict[str, str]:
    latest: dict[str, tuple[int, str]] = {}
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = _parse_frontmatter(raw)
        name, version = _name_version_from_path(path, meta)
        if not name or not version.startswith("v"):
            continue
        try:
            number = int(version[1:])
        except ValueError:
            number = -1
        if name not in latest or number > latest[name][0]:
            latest[name] = (number, version)
    return {name: version for name, (_n, version) in latest.items()}


def _prompt_record(path: Path, latest_by_name: dict[str, str]) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        stat = path.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"no pude leer {path.name}: {exc}") from exc
    meta, body = _parse_frontmatter(raw)
    name, version = _name_version_from_path(path, meta)
    group = path.parent.name
    kind = str(meta.get("kind") or ("rule" if group == "rules" else "intent"))
    deprecated = bool(meta.get("deprecated"))
    latest = latest_by_name.get(name) == version if version else False
    status = "deprecated" if deprecated else ("latest" if latest or group == "rules" else "legacy")
    info = _prompt_info(name, status, meta, body)
    return {
        "id": _repo_rel(path),
        "path": _repo_rel(path),
        "name": name,
        "version": version,
        "group": group,
        "kind": kind,
        "status": status,
        "latest": latest,
        "deprecated": deprecated,
        "superseded_by": meta.get("superseded_by") or "",
        "includes": meta.get("includes") or [],
        "purpose": _purpose(meta, body),
        **info,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "line_count": raw.count("\n") + 1,
        "editable": True,
    }


def _list_prompt_files() -> list[Path]:
    if not _PROMPTS_ROOT.exists():
        return []
    return sorted(path for path in _PROMPTS_ROOT.rglob("*.md") if path.is_file())


def prompts_page(static_dir: Path) -> FileResponse:
    return FileResponse(static_dir / "prompts.html")


def api_prompts_list() -> dict[str, Any]:
    files = _list_prompt_files()
    latest_by_name = _latest_versions([p for p in files if p.parent.name == "intents"])
    records = [_prompt_record(path, latest_by_name) for path in files]
    inline = [_inline_record(item) for item in _INLINE_PROMPTS]
    return {
        "root": _repo_rel(_PROMPTS_ROOT),
        "count": len(records),
        "prompts": records,
        "inline": inline,
    }


def _inline_record(item: dict[str, Any]) -> dict[str, Any]:
    importance = str(item.get("importance") or "medium")
    if importance not in _IMPORTANCE_LABELS:
        importance = "medium"
    impact = str(item.get("impact") or "Afecta una superficie inline del sistema.").strip()
    return {
        **item,
        "group": "inline",
        "kind": "inline",
        "status": "source",
        "latest": False,
        "deprecated": False,
        "includes": [],
        "importance": importance,
        "importance_label": _IMPORTANCE_LABELS[importance],
        "importance_reason": impact,
        "effective": str(item.get("effective") or item.get("purpose") or "").strip(),
        "impact": impact,
        "editable": False,
    }


def _inline_item_by_id(prompt_id: str) -> dict[str, Any]:
    for item in _INLINE_PROMPTS:
        if item["id"] == prompt_id:
            return item
    raise HTTPException(status_code=404, detail="prompt inline no encontrado")


def _safe_source_path(rel_path: str) -> Path:
    rel = (rel_path or "").strip().lstrip("/")
    path = (_REPO_ROOT / rel).resolve()
    try:
        path.relative_to(_REPO_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path fuera del repo") from exc
    if path.suffix != ".py":
        raise HTTPException(status_code=400, detail="solo source Python")
    return path


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def _literal_string(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except Exception:
        value = None
    if isinstance(value, str):
        return value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _symbol_patterns(symbol: str) -> list[str]:
    parts = symbol.replace(" / ", "/").split("/")
    return [part.strip() for part in parts if part.strip()]


def _symbol_matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _extract_inline_content(item: dict[str, Any], source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HTTPException(status_code=500, detail=f"no pude parsear source: {exc}") from exc
    patterns = _symbol_patterns(str(item.get("symbol") or ""))
    wanted_line = item.get("line")
    sections: list[str] = []
    for node in ast.walk(tree):
        name = ""
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _target_names(node)
            name = next((candidate for candidate in names if _symbol_matches(candidate, patterns)), "")
        if not name or not _symbol_matches(name, patterns):
            continue
        if wanted_line and getattr(node, "lineno", None) != wanted_line:
            continue
        body = None
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            body = _literal_string(node.value)
        if body is None:
            body = ast.get_source_segment(source, node)
        if body:
            sections.append(f"# {name} · line {getattr(node, 'lineno', '?')}\n\n{body.strip()}")
    if sections:
        return "\n\n# ---\n\n".join(sections).strip() + "\n"
    return "No pude extraer el contenido estático de este prompt. Revisá el source indicado arriba.\n"


def api_inline_prompt_read(id: str) -> dict[str, Any]:
    item = _inline_item_by_id(id)
    path = _safe_source_path(str(item.get("path") or ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="source no encontrado")
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"no pude leer source: {exc}") from exc
    return {
        "prompt": _inline_record(item),
        "content": _extract_inline_content(item, source),
    }


def api_prompt_read(path: str) -> dict[str, Any]:
    prompt_path = _safe_prompt_path(path)
    if not prompt_path.exists():
        raise HTTPException(status_code=404, detail="prompt no encontrado")
    try:
        content = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"no pude leer el prompt: {exc}") from exc
    latest_by_name = _latest_versions(_list_prompt_files())
    return {
        "prompt": _prompt_record(prompt_path, latest_by_name),
        "content": content,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            if content and not content.endswith("\n"):
                fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def api_prompt_write(req: PromptWriteRequest) -> dict[str, Any]:
    prompt_path = _safe_prompt_path(req.path)
    if not prompt_path.exists():
        raise HTTPException(status_code=404, detail="prompt no encontrado")
    content = req.content.replace("\r\n", "\n").replace("\r", "\n")
    if not content.strip():
        raise HTTPException(status_code=400, detail="contenido vacío")
    try:
        _atomic_write_text(prompt_path, content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"no pude guardar el prompt: {exc}") from exc
    latest_by_name = _latest_versions(_list_prompt_files())
    return {
        "ok": True,
        "prompt": _prompt_record(prompt_path, latest_by_name),
    }


def register_prompt_routes(app, static_dir: Path, require_admin_token) -> dict[str, object]:
    static_dir = Path(static_dir)

    def page() -> FileResponse:
        return prompts_page(static_dir)

    app.get("/prompts")(page)
    app.get("/api/prompts", dependencies=[Depends(require_admin_token)])(api_prompts_list)
    app.get("/api/prompts/file", dependencies=[Depends(require_admin_token)])(api_prompt_read)
    app.get("/api/prompts/inline", dependencies=[Depends(require_admin_token)])(api_inline_prompt_read)
    app.post("/api/prompts/file", dependencies=[Depends(require_admin_token)])(api_prompt_write)

    return {
        "prompts_page": page,
        "api_prompts_list": api_prompts_list,
        "api_prompt_read": api_prompt_read,
        "api_inline_prompt_read": api_inline_prompt_read,
        "api_prompt_write": api_prompt_write,
    }
