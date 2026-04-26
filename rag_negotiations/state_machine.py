"""State machine de las negociaciones del WA Auto-Pilot.

Cada negociación atraviesa un grafo dirigido pequeño. El propósito de este
módulo es:

1. **Hacer explícito** qué transiciones son legales (vs un campo `status`
   suelto que cualquiera podría sobreescribir con cualquier string).
2. **Centralizar** el grafo en un solo dict para que el dashboard, el
   orchestrator (Fase 3) y el CLI muestren la misma realidad.
3. **Quedarse a nivel de datos** — Fase 0 NO incluye lógica del classifier
   ni LLM calls. Solo valida transiciones y devuelve el estado siguiente.

Estados (8 totales, 4 terminales):

```
                         ┌─────────────────────────────────────┐
                         ▼                                     │
[draft] ──launch─→ [launched] ──first_msg_ack─→ [in_flight] ───┘
                         │                            │
                  user_cancels                  classifier_low_conf
                         │                            ▼
                         │                    [escalated] ──user_resumes─→ [in_flight]
                         │                            │
                         │                     user_takes_over
                         ▼                            ▼
                  [cancelled]                 [closed_ok | closed_fail]

                         ▲
                  perimeter_violation       (also de in_flight, escalated)
                         │
                  [out_of_perimeter]
```

- **`draft`** — el sistema generó un primer mensaje + el user todavía no
  aprobó.
- **`launched`** — el user aprobó. El primer mensaje fue enviado al
  bridge pero todavía no hay respuesta de la otra parte.
- **`in_flight`** — la otra parte respondió al menos una vez. El bot
  está handleando autonomamente.
- **`escalated`** — el classifier rebajó la confidence < threshold y
  el user fue notificado para responder a mano. Vuelve a `in_flight`
  si el user aprueba la sugerencia o manda su propia respuesta.
- **`closed_ok`** — terminada con acuerdo (agreement detectado).
- **`closed_fail`** — terminada sin acuerdo (rejection o timeout).
- **`cancelled`** — el user canceló manualmente.
- **`out_of_perimeter`** — la otra parte se salió del scope acordado
  al lanzar (ej. quiso negociar un tema distinto). Terminal — el user
  retoma manualmente sin que el bot moleste.

Estados terminales (no tienen transiciones salientes):
`closed_ok`, `closed_fail`, `cancelled`, `out_of_perimeter`.

NOTA (Fase 0): este módulo es **puro** — no toca DB. La capa de
persistence vive en `rag_negotiations.crud`. El orchestrator de Fase 3
va a invocar `transition()` para validar antes de hacer UPDATE.
"""
from __future__ import annotations

from typing import Final


#: Orden topológico de los estados — útil para UIs que ordenan tablas
#: por progreso, o para queries SQL "negociaciones más adelantadas
#: primero".
STATE_ORDER: Final[tuple[str, ...]] = (
    "draft",
    "launched",
    "in_flight",
    "escalated",
    "closed_ok",
    "closed_fail",
    "cancelled",
    "out_of_perimeter",
)


#: Estados terminales — sin transiciones salientes. El orchestrator NO
#: debe llamar `transition()` cuando una negociación está en uno de
#: estos. UI muestra como "cerrada" / "archivada".
TERMINAL_STATES: Final[frozenset[str]] = frozenset({
    "closed_ok",
    "closed_fail",
    "cancelled",
    "out_of_perimeter",
})


#: Cierres "exitosos" para métricas de success rate. `out_of_perimeter`
#: NO cuenta — es un cierre prematuro por scope inválido, no por éxito
#: del bot.
CLOSED_OK_STATES: Final[frozenset[str]] = frozenset({"closed_ok"})


#: Grafo de transiciones legales: `{estado_origen: {transición: estado_destino}}`.
#: Las transiciones son nombres descriptivos del trigger humano-leíble
#: para que dashboards puedan filtrar tipo "mostrame todos los
#: `user_cancels` del último mes".
VALID_TRANSITIONS: Final[dict[str, dict[str, str]]] = {
    "draft": {
        # User aprobó el primer mensaje en la pre-launch screen. El
        # mensaje sale al bridge, esperamos primera respuesta.
        "launch": "launched",
        # User cerró la PWA sin aprobar / clickeó cancelar.
        "user_cancels": "cancelled",
    },
    "launched": {
        # La otra parte respondió. El bot empieza a handlear.
        "first_msg_ack": "in_flight",
        # User cancela post-launch (ej. se arrepintió antes de que la
        # otra parte respondiera).
        "user_cancels": "cancelled",
        # Timeout: la otra parte nunca respondió en N días. Cierra
        # como fail por silencio. `closed_fail` permite la métrica
        # "ghosting rate".
        "timeout": "closed_fail",
    },
    "in_flight": {
        # Classifier rebajó confianza — escalamos al user.
        "classifier_low_conf": "escalated",
        # Classifier detectó acuerdo + side effects ya disparados.
        "agreement_detected": "closed_ok",
        # Classifier detectó rechazo claro de la otra parte.
        "rejection_detected": "closed_fail",
        # Otra parte se fue de scope (quiso meter tema fuera del
        # perímetro acordado al lanzar). El user retoma manualmente.
        "perimeter_violation": "out_of_perimeter",
        # User intervino manualmente vía PWA / WA — el bot pausa en
        # silencio. Cancelled porque el bot ya no tiene autorización.
        "user_takes_over": "cancelled",
        # Hard timeout (ej. >7 días sin actividad) → cerramos como
        # fail. El orchestrator dispara esto vía cron / fsevents.
        "timeout": "closed_fail",
    },
    "escalated": {
        # User mandó su respuesta — el bot vuelve a handlear los
        # próximos turnos.
        "user_resumes": "in_flight",
        # User decidió cerrar manualmente la negociación con la
        # respuesta que dio (ej. "no, gracias"). Bot no toca más.
        "user_takes_over": "cancelled",
        # User responde + el classifier detecta que esa respuesta
        # fue el cierre de la negociación.
        "agreement_detected": "closed_ok",
        "rejection_detected": "closed_fail",
        # Misma puerta de salida que `in_flight`.
        "perimeter_violation": "out_of_perimeter",
    },
    # Estados terminales — sin salidas.
    "closed_ok": {},
    "closed_fail": {},
    "cancelled": {},
    "out_of_perimeter": {},
}


class InvalidTransitionError(ValueError):
    """Raised cuando se pide una transición que no existe en el grafo.

    Mensaje incluye `from_state`, `transition`, y la lista de
    transiciones legales desde ese estado para debugging rápido.
    """


def is_terminal(state: str) -> bool:
    """True si el estado es terminal (sin transiciones salientes)."""
    return state in TERMINAL_STATES


def legal_transitions(from_state: str) -> dict[str, str]:
    """Devuelve `{transición: estado_destino}` desde `from_state`.

    Para estados terminales devuelve dict vacío. Estados desconocidos
    también devuelven dict vacío (callers que validan con `if not
    legal_transitions(...)` quedan cubiertos en ambos casos).
    """
    return dict(VALID_TRANSITIONS.get(from_state, {}))


def can_transition(from_state: str, transition: str) -> bool:
    """True si `transition` es legal desde `from_state`."""
    return transition in VALID_TRANSITIONS.get(from_state, {})


def transition(from_state: str, transition_name: str) -> str:
    """Aplica `transition_name` y devuelve el `to_state`.

    Args:
        from_state: estado actual de la negociación.
        transition_name: nombre del trigger (ver `VALID_TRANSITIONS`).

    Returns:
        El estado siguiente.

    Raises:
        InvalidTransitionError: si la transición no existe desde
            `from_state`. Incluye en el mensaje las legales para
            que el caller pueda corregir.
    """
    legal = VALID_TRANSITIONS.get(from_state, {})
    if transition_name not in legal:
        legal_names = sorted(legal.keys())
        if not legal_names:
            hint = (
                f"`{from_state}` es estado terminal (sin transiciones "
                "salientes)."
            )
        else:
            hint = f"transiciones legales: {legal_names}"
        raise InvalidTransitionError(
            f"transición `{transition_name}` inválida desde "
            f"`{from_state}`. {hint}"
        )
    return legal[transition_name]


__all__ = [
    "CLOSED_OK_STATES",
    "InvalidTransitionError",
    "STATE_ORDER",
    "TERMINAL_STATES",
    "VALID_TRANSITIONS",
    "can_transition",
    "is_terminal",
    "legal_transitions",
    "transition",
]
