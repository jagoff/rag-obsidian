#!/usr/bin/env python3
"""Adversarial query generator para eval del retriever (skill: evaluate-rag).

Process (Hamel Husain, evals-skills@evaluate-rag):
  1. Sample target chunks A con un fact extraíble.
  2. Para cada A: encontrar B, C (top-k similares en sqlite-vec, ≠ A,
     idealmente de notas distintas — no son chunks-hermanos del mismo archivo).
  3. Llamar al helper LLM (qwen2.5:3b deterministic): generar query usando
     terminología compartida B+C que SOLO A puede responder.
  4. Filtrar por realismo (helper Likert 1-5): keep si score ≥ threshold.
  5. Output queries_adversarial.yaml en el mismo shape que queries.yaml
     (singles + chains keys). NO toca queries.yaml hand-curated.

Run:
    .venv/bin/python scripts/generate_adversarial_queries.py \\
        --n-targets 100 \\
        --output queries_adversarial.yaml

Eval (después):
    rag eval --file queries_adversarial.yaml --latency
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Garantizar import de `rag` cuando se corre `.venv/bin/python scripts/foo.py`
# desde el repo root (sys.path[0] queda en scripts/, no en cwd).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

# Lazy import (rag carga sentence-transformers + MLX al import; evitamos
# en CLI parse).
def _lazy_imports():
    import rag
    return rag


# ─────────────────────────────────────────────────────────────────────────
# DECISIÓN 1 (USER) — Target chunk filter
# ─────────────────────────────────────────────────────────────────────────
# ¿Qué chunk del vault califica como "target A" (tiene un fact extraíble
# que sirve de gold answer)? Domain knowledge tuya: conocés tu vault.
#
# Ejemplos de filtros plausibles:
#   - Mínimo de chars (chunks <200 chars probablemente no tienen fact)
#   - Excluir folders (00-Inbox/conversations, 99-Mentions, 99-AI/)
#     porque son ruido / chats / system artifacts
#   - Pedir presencia de ≥1 número o fecha (heurística "tiene un dato duro")
#   - Pedir que no sea daily note (`\d{4}-\d{2}-\d{2}\.md` filename)
#
# Devolvé True para keep, False para skip.
# ─────────────────────────────────────────────────────────────────────────
_EXCLUDED_PREFIXES = (
    "00-Inbox/conversations/",   # chats con el LLM, no hechos del user
    "99-Mentions/",               # auto-generated mention notes
    "99-obsidian/",  # system artifacts (99-AI/, plans, specs)
)
_DAILY_NOTE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\.md$")


def is_target_candidate(chunk_text: str, meta: dict) -> bool:
    """Filter: chunk con fact extraíble.

    Reject:
      - Cross-source ingesters (reminders://, gmail://, etc.) — no son
        notas vault-relative y `expected` queda inválido para el eval.
      - Folders system / chats / mentions (no son knowledge canónico)
      - Daily notes (texto stream-of-thought, fact mal definido)
      - Chunks <250 chars (probablemente truncated o título solo)
    """
    if len(chunk_text.strip()) < 250:
        return False
    fpath = (meta.get("file") or "").lstrip("/")
    if "://" in fpath:  # reminders://, gmail://, calendar://, etc.
        return False
    if any(fpath.startswith(p) for p in _EXCLUDED_PREFIXES):
        return False
    if _DAILY_NOTE_RE.search(fpath):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────
# DECISIÓN 2 (USER) — Adversarial prompt
# ─────────────────────────────────────────────────────────────────────────
# Este prompt es el corazón del feature. El LLM tiene que generar una query
# que use terminología compartida con los distractors B/C pero que SOLO el
# chunk A pueda responder con confianza.
#
# Notas para el prompt:
#   - El idioma del vault es ES rioplatense + inglés (technical). El user
#     consulta en castellano informal (voseo). El prompt debería pedir
#     queries con esa cadencia, no formal ("¿Cuándo ocurrió X?" — no
#     "Determine the date of X").
#   - Output debe ser JSON estricto: {"question": "..."} para parse seguro
#     (HELPER_OPTIONS ya es deterministic temp=0).
#   - Decile al modelo "no incluyas el título de A en la query" (anti-leak
#     BM25, igual regla que queries.yaml).
#   - Decile que la pregunta tenga 5-15 palabras, no más (real user style).
#   - Mostrale los 3 chunks (A, B, C) como bloques delimitados, igual que
#     `_wrap_untrusted` hace para no-injection.
# ─────────────────────────────────────────────────────────────────────────
def build_adversarial_prompt(chunk_a: str, chunk_b: str, chunk_c: str) -> str:
    """Genera la query adversarial con JSON output estricto.

    Few-shot calibrado con queries.yaml real: estilo TOPIC (corto, sin "qué
    pasó con", a veces sin verbo). El modelo reproduce el estilo del few-shot
    en lugar de inventarse "cuándo qué pasó con dónde cómo" Frankensteins.
    """
    return f"""\
Generás queries adversariales para un sistema de retrieval. Te paso 3 chunks
de un vault Obsidian. Tu trabajo: escribir UNA query corta que el usuario
real (Fer) podría tipear, que SOLO el chunk A responde con confianza, pero
que use terminología compartida con los distractors B y C.

Estilo del usuario real:
  - 3 a 10 palabras MAX (no 15+).
  - Lowercase, sin "?", sin signo de exclamación.
  - Topic-style: a veces sólo sustantivos + tema ("axe fx 3 configuración"),
    sin verbo ni interrogativo. Otras veces arranca con UNA palabra
    interrogativa ("qué", "cómo", "por qué", "cuándo") — NUNCA varias
    seguidas.
  - Voseo informal cuando hay persona ("tenés", "subiste"), no "tu" formal.
  - PROHIBIDO: "qué pasó con", "qué onda con", "qué hay sobre" — suena a IA.
  - PROHIBIDO: copiar literal los temas que te muestro abajo — armá una
    query NUEVA basada en los chunks A/B/C de abajo, no en estos ejemplos.

Ejemplos SÓLO para calibrar tono (NO copiar contenido):
  · "comunicación no violenta"
  · "cómo manejar el estrés"
  · "por qué hyde falla con modelos chicos"

CHUNK A (fuente del fact — la query DEBE poder responderse desde acá):
<<<A
{chunk_a[:1200]}
>>>A

CHUNK B (distractor cercano, NO responde la query):
<<<B
{chunk_b[:1200]}
>>>B

CHUNK C (otro distractor):
<<<C
{chunk_c[:1200]}
>>>C

Reglas duras:
  1. Usá terminología compartida entre B y C — eso hace la query adversarial.
  2. SOLO A debe poder responderla con el dato específico.
  3. NO incluyas el título literal del chunk A en la query (anti-BM25 leak).
  4. Imitá el estilo del few-shot de arriba (corto, topic-style, lowercase).

Devolvé EXCLUSIVAMENTE JSON, sin markdown, sin explicaciones:
{{"question": "..."}}"""


# ─────────────────────────────────────────────────────────────────────────
# DECISIÓN 3 (USER) — Realism threshold
# ─────────────────────────────────────────────────────────────────────────
# El skill sugiere 4-5 sobre escala 1-5. Tu vault tiene queries reales bien
# concretas (ver queries.yaml singles); los sintéticos suelen sonar más
# formales. Threshold posibles:
#   - 4 (estricto, dataset chico de alta calidad)
#   - 3 (permisivo, más cobertura, más ruido)
# ─────────────────────────────────────────────────────────────────────────
REALISM_MIN_SCORE = 2  # qwen2.5:3b es brutal; threshold 2 recupera ~20pp falsos negativos vistos en N=100


REALISM_PROMPT_TEMPLATE = """\
Calificá del 1 al 5 cuán natural suena esta query como pregunta de un usuario
real al motor de búsqueda de su vault Obsidian.

Anchors (estudialos antes de calificar):

  5 (idéntico a queries reales del usuario):
    · "comunicación no violenta"
    · "axe fx 3 configuración"
    · "shortcuts de macos"
    · "cómo manejar el estrés en el trabajo"
    · "por qué hyde falla con modelos chicos"

  3 (plausible, podría tipearla):
    · "rutinas saludables para reducir estrés"
    · "marketing para redes sociales"
    · "qué hacer cuando hay errores de aws"

  1 (claramente sintético / formal manual):
    · "Determine the optimal configuration for the Axe FX III device"
    · "Cuál es el procedimiento estándar para gestionar el estrés laboral"
    · "Por favor explíqueme detalladamente cómo funciona HyDE"

Query a calificar:
"{question}"

Respondé SOLO con un dígito: 1, 2, 3, 4 ó 5."""


def call_helper(prompt: str, max_tokens: int = 200) -> str:
    """Call al helper LLM (qwen2.5:3b deterministic, MLX backend).

    Mismo path que reformulate_query / get_context_summary etc.
    """
    rag = _lazy_imports()
    resp = rag._helper_client().chat(
        model=rag.HELPER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={**rag.HELPER_OPTIONS, "num_predict": max_tokens},
        keep_alive=rag.LLM_KEEP_ALIVE,
    )
    return resp.message.content.strip()


def parse_question_json(raw: str) -> str | None:
    """Extraer 'question' del output JSON del helper. None si no parsea."""
    raw = raw.strip()
    # Tolerate ```json fences
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        d = json.loads(raw)
        q = d.get("question")
        if isinstance(q, str) and q.strip():
            return q.strip().rstrip("?")
    except json.JSONDecodeError:
        pass
    return None


def parse_realism_score(raw: str) -> int | None:
    """Extraer dígito 1-5 del output del helper."""
    for ch in raw.strip():
        if ch in "12345":
            return int(ch)
    return None


def sample_targets(rag, n: int, seed: int = 42) -> list[dict]:
    """Pull all chunks, filter por is_target_candidate, sample n random."""
    col = rag.get_db()
    if col.count() == 0:
        sys.exit("Index vacío — corré `rag index` primero.")

    bag = col.get(include=["documents", "metadatas"])
    pool: list[dict] = []
    for cid, doc, meta in zip(bag["ids"], bag["documents"], bag["metadatas"]):
        if not doc or not meta:
            continue
        if is_target_candidate(doc, meta):
            pool.append({"chunk_id": cid, "doc": doc, "meta": meta})

    print(f"[targets] pool tras filter: {len(pool)} chunks", file=sys.stderr)
    if len(pool) < n:
        print(f"[targets] solo {len(pool)} candidates, devuelvo todos",
              file=sys.stderr)
        return pool

    rng = random.Random(seed)
    return rng.sample(pool, n)


def find_distractors(rag, target: dict, exclude_file: str) -> tuple[str, str] | None:
    """Top-2 nearest neighbors de target (≠ mismo file).

    Returns (chunk_b_text, chunk_c_text) o None si no hay 2 distractors
    de archivos distintos.
    """
    qemb = rag.embed([target["doc"]])
    res = rag.get_db().query(
        query_embeddings=qemb,
        n_results=10,  # buscamos 10 para tener slack tras filtrar same-file
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    distractors: list[str] = []
    for d, m in zip(docs, metas):
        if not d or not m:
            continue
        if m.get("file") == exclude_file:
            continue  # same-file chunks no son distractors útiles
        distractors.append(d)
        if len(distractors) >= 2:
            break

    if len(distractors) < 2:
        return None
    return distractors[0], distractors[1]


def generate_one(rag, target: dict) -> tuple[dict | None, str]:
    """Pipeline completo para un target: distractors → query → realism filter.

    Returns (dict | None, reason_str). reason útil para diagnosticar el
    primer batch ("no_distractors" / "json_parse" / "low_realism" / "ok").
    """
    exclude_file = target["meta"].get("file") or ""
    pair = find_distractors(rag, target, exclude_file)
    if pair is None:
        return None, "no_distractors"
    chunk_b, chunk_c = pair

    prompt = build_adversarial_prompt(target["doc"], chunk_b, chunk_c)
    raw = call_helper(prompt, max_tokens=200)
    question = parse_question_json(raw)
    if question is None:
        return None, f"json_parse(raw={raw[:80]!r})"

    # Rule-based filter (reemplaza LLM judge — qwen2.5:3b es bipolar 1/5).
    # Heurísticas calibradas sobre queries.yaml real:
    #   - 3-12 palabras (real range: 1-9, dejamos margen)
    #   - sin punctuation formal (`:`, `;`, `…`, `?` final no cuenta)
    #   - lowercase first char (real queries no son tipo manual)
    rejected = _rule_based_realism_reject(question)
    if rejected is not None:
        return None, f"rule_reject({rejected}) q={question[:80]!r}"

    # Self-validation gate: verificar que el target chunk APAREZCA en top-20
    # cuando re-corremos retrieve() sobre la query generada. Filtra labeling
    # noise — caso típico: "rutinas saludables para reducir estrés" generada
    # del chunk Moka-Pasantes-Catolica donde se menciona estrés en passing,
    # pero el verdadero canonical answer en el vault es Charla-Ansiedad.
    #
    # Threshold k=20 (NO k=3): k=3 era selection bias — solo queries que YA
    # rankean top-3 con multi_query=False pasaban, ceiling effect en eval
    # downstream (medido 97.83% hit@5 con k=3 gate). k=20 valida que la
    # query ES semánticamente sobre el target (filtra noise) pero deja casos
    # rankeo-difíciles donde la posición 4-20 es justo lo que el reranker
    # del eval debe resolver — eso evalúa el ranker, no el embedder.
    if not _target_in_topk(rag, question, exclude_file, k=20):
        return None, f"target_not_in_top20 q={question[:80]!r}"

    return {
        "question": question.lower(),
        "expected": [exclude_file],
    }, "ok"


def _target_in_topk(rag, question: str, target_file: str, k: int = 3) -> bool:
    """Re-retrieve la query y verificar que `target_file` aparezca en top-k.

    Usa retrieve() con defaults idénticos al eval (k=10 para tener slack,
    pero check si target está en los primeros k=3). Multi-query OFF para
    velocidad — sola la query original. HyDE OFF (default).
    """
    try:
        result = rag.retrieve(
            col=rag.get_db(),
            question=question,
            k=10,
            folder=None,
            multi_query=False,
            auto_filter=False,
            caller="adv_gen_validate",
        )
    except Exception as exc:
        print(f"[validate] retrieve failed: {exc}", file=sys.stderr)
        return False
    metas = result.get("metas") or []
    for m in metas[:k]:
        if (m or {}).get("file") == target_file:
            return True
    return False


def _rule_based_realism_reject(q: str) -> str | None:
    """Returns None si query pasa, str con razón si rejected."""
    qs = q.strip()
    if not qs:
        return "empty"
    word_count = len(qs.split())
    if word_count < 3:
        return f"too_short({word_count}w)"
    if word_count > 12:
        return f"too_long({word_count}w)"
    body = qs.rstrip("?")
    for ch in (":", ";", "…"):
        if ch in body:
            return f"formal_punct({ch})"
    # Detectar formalismos manual-style
    formal_markers = ("Por favor", "Determinar", "Cuál es el procedimiento",
                      "Explique", "Detalle", "Indique")
    for marker in formal_markers:
        if marker.lower() in qs.lower():
            return f"formal_marker({marker})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-targets", type=int, default=100,
                    help="Cantidad de target chunks a samplear (default 100)")
    ap.add_argument("--output", default="queries_adversarial.yaml",
                    help="Path YAML de salida")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rag = _lazy_imports()
    targets = sample_targets(rag, args.n_targets, seed=args.seed)
    print(f"[main] generando adversarials sobre {len(targets)} targets",
          file=sys.stderr)

    out_queries: list[dict] = []
    for i, t in enumerate(targets, 1):
        try:
            item, reason = generate_one(rag, t)
        except Exception as e:
            print(f"[{i}/{len(targets)}] ✗ exception: {e}", file=sys.stderr)
            continue
        if item is None:
            print(f"[{i}/{len(targets)}] ✗ {reason}", file=sys.stderr)
            continue
        item.pop("_score", None)
        out_queries.append(item)
        print(f"[{i}/{len(targets)}] ✓ {item['question'][:60]}",
              file=sys.stderr)

    out_path = Path(args.output)
    out_path.write_text(
        yaml.safe_dump(
            {"queries": out_queries, "chains": []},
            allow_unicode=True, sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[done] {len(out_queries)} queries adversariales → {out_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
