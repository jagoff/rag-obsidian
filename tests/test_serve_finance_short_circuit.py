"""Tests para el short-circuit de finance/tarjetas en `rag serve /query`.

Origen: 2026-04-26 user report — el listener de WhatsApp pega al endpoint
`rag serve /query` (puerto 7832, NO al `/api/chat` del web). Ese endpoint
no tenía el pre-router de tools, así que queries como "cuánto gasté de
tarjeta este mes" caían al retrieve genérico y traían 90 notas random
del vault. El LLM final alucinaba con montos del Gmail digest o FinOps
reports tangenciales.

Este archivo cubre:
  1. `_is_finance_or_cards_query` — el detector regex que decide si
     dispara el short-circuit.
  2. `_format_finance_cards_block` — render determinístico de los datos
     pegados al prompt del LLM helper.

`_finance_cards_comment` requiere una llamada al LLM helper (qwen2.5:3b
via ollama) y por lo tanto solo se prueba via integration en el
endpoint smoke test (que requiere el daemon `rag serve` corriendo).
"""
from __future__ import annotations

import pytest

import rag


# ── 1. _is_finance_or_cards_query ───────────────────────────────────────────


_POSITIVE_CASES = [
    # Marcas + tarjetas
    "cuánto debo de la visa?",
    "saldo a pagar de mi tarjeta",
    "cuándo vence mi visa?",
    "fecha de cierre de la mastercard",
    "resumen de tarjeta amex",
    "cuánto debo este mes?",
    # MOZE / gastos
    "cuánto gasté este mes",
    "presupuesto del mes",
    "mostrame los gastos",
    "finanzas del mes",
    "moze",
    # Mixtos (disparan ambos)
    "decime cuanto gaste de tarjeta este mes?",
    "cuál fue mi último gasto en la visa",
    # Naturales (post-fix)
    "últimos consumos",
    "movimientos de la tarjeta",
    "últimas compras",
    "recientes movimientos",
    "mi crédito",
]

_NEGATIVE_CASES = [
    # Weather
    "cómo está el clima",
    "va a llover mañana?",
    # Metachat
    "hola",
    "gracias",
    "qué podés hacer",
    # Tasks
    "qué tengo esta semana",
    "qué hay mañana",
    # Queries sobre vault que coinciden tangencialmente
    "tengo una nota sobre coaching",
    "buscar notas sobre python",
    # Demasiado largo (>16 tokens) — el classifier corta para evitar
    # falsos positivos en queries narrativas. Token boundary aproximado.
    "ayer pasamos un día tranquilo en casa con la familia tomando mate y "
    "charlando largo rato sobre las vacaciones del año pasado",
]


@pytest.mark.parametrize("query", _POSITIVE_CASES)
def test_finance_query_classifier_matches(query: str):
    assert rag._is_finance_or_cards_query(query) is True, (
        f"Query {query!r} debería disparar el short-circuit de finance/cards"
    )


@pytest.mark.parametrize("query", _NEGATIVE_CASES)
def test_finance_query_classifier_skips(query: str):
    assert rag._is_finance_or_cards_query(query) is False, (
        f"Query {query!r} NO debería disparar el short-circuit"
    )


def test_finance_query_classifier_handles_empty():
    assert rag._is_finance_or_cards_query("") is False
    assert rag._is_finance_or_cards_query("   ") is False
    assert rag._is_finance_or_cards_query(None) is False  # type: ignore[arg-type]


# ── 2. _format_finance_cards_block ──────────────────────────────────────────


def test_format_block_with_both_finance_and_cards():
    finance = {
        "month_label": "2026-04",
        "days_elapsed": 26,
        "days_in_month": 30,
        "ars": {
            "this_month": 5_119_708.68,
            "prev_month": 3_351_400.00,
            "delta_pct": 52.8,
            "projected": 5_907_355.79,
            "top_categories": [
                {"name": "House", "amount": 2_785_000.00, "share": 0.5440},
                {"name": "Maria", "amount": 700_000.00, "share": 0.1367},
            ],
        },
        "usd": {"this_month": 145.50},
    }
    cards = [
        {
            "brand": "Visa",
            "last4": "1059",
            "total_ars": 549_438.75,
            "total_usd": 98.93,
            "due_date": "2026-04-08",
            "closing_date": "2026-03-26",
            "top_purchases_ars": [
                {"description": "Merpago*idilicadeco", "amount": 153_333.33},
                {"description": "Pago tic ppt adventistas", "amount": 122_550.00},
            ],
            "top_purchases_usd": [
                {"description": "Apple.com/bill", "amount": 24.99},
                {"description": "Claude.ai subscr in1t", "amount": 20.00},
            ],
        },
    ]

    block = rag._format_finance_cards_block(finance, cards)

    # Both sections present
    assert "### Tarjetas" in block
    assert "### Gastos del mes" in block
    # Card data
    assert "Visa ····1059" in block
    assert "$549.439" in block or "$549.438" in block  # rounding tolerance
    assert "U$S98.93" in block
    assert "Vence: 2026-04-08" in block
    assert "Merpago*idilicadeco" in block
    assert "Apple.com/bill" in block
    # MOZE data
    assert "2026-04" in block
    assert "$5.119.709" in block or "$5.119.708" in block
    assert "House" in block and "54%" in block
    assert "U$S145.50" in block


def test_format_block_with_only_cards():
    cards = [{
        "brand": "Visa", "last4": "1059",
        "total_ars": 100_000.0, "due_date": "2026-04-08",
    }]
    block = rag._format_finance_cards_block(None, cards)
    assert "### Tarjetas" in block
    assert "### Gastos del mes" not in block
    assert "Visa ····1059" in block


def test_format_block_with_only_finance():
    finance = {
        "month_label": "2026-04",
        "days_elapsed": 10, "days_in_month": 30,
        "ars": {"this_month": 100_000.0, "top_categories": []},
        "usd": {},
    }
    block = rag._format_finance_cards_block(finance, None)
    assert "### Gastos del mes" in block
    assert "### Tarjetas" not in block


def test_format_block_empty_input_returns_empty():
    assert rag._format_finance_cards_block(None, None) == ""
    assert rag._format_finance_cards_block(None, []) == ""
    assert rag._format_finance_cards_block({}, []) == ""


def test_format_block_handles_missing_optional_fields():
    """Cards con minimal fields (sin top_purchases, sin closing_date) no rompen."""
    cards = [{"brand": "Visa", "last4": "1059", "total_ars": 100_000.0}]
    block = rag._format_finance_cards_block(None, cards)
    assert "Visa ····1059" in block
    # No crash — campos faltantes se omiten silenciosamente


# ── 3. _finance_cards_comment fallback ──────────────────────────────────────


def test_comment_returns_canned_msg_when_no_data():
    """Sin finance ni cards, el helper NO llama al LLM (block vacío) y
    devuelve el mensaje canned. Comportamiento crítico: aún cuando ollama
    está caído, el endpoint no devuelve respuesta vacía."""
    answer = rag._finance_cards_comment("cuánto debo?", None, None)
    assert "No tengo data fresca" in answer
    assert "tarjetas" in answer.lower() or "moze" in answer.lower()
