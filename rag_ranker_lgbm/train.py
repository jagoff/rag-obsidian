"""Training pipeline para LightGBM lambdarank.

Wrapper sobre `lgb.LGBMRanker` con hyperparams sensibles para el corpus
del usuario. Persiste el modelo a disk + métricas a `rag_tune` (mismo
patrón que `rag tune --apply`).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _default_model_path() -> Path:
    """Default location: same dir as ranker.json y otros artifacts ML."""
    return (
        Path.home() / ".local/share/obsidian-rag/ranker.lgbm"
    )


# Re-exported para CLI / consumers.
DEFAULT_MODEL_PATH = _default_model_path()


def _validation_split(
    X: list[list[float]],
    y: list[int],
    group: list[int],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Split por GRUPO (no por candidate) — mantiene los candidatos de cada
    query juntos en train o val.

    Lambdarank con val_fraction=0 (sin validation) sigue funcionando — no
    early stopping pero tampoco overfit signal.
    """
    import random

    n_groups = len(group)
    if n_groups < 5 or val_fraction <= 0:
        return {
            "X_train": X,
            "y_train": y,
            "group_train": group,
            "X_val": [],
            "y_val": [],
            "group_val": [],
        }

    rng = random.Random(seed)
    indices = list(range(n_groups))
    rng.shuffle(indices)
    n_val = max(1, int(n_groups * val_fraction))
    val_indices = set(indices[:n_val])

    X_train: list[list[float]] = []
    y_train: list[int] = []
    group_train: list[int] = []
    X_val: list[list[float]] = []
    y_val: list[int] = []
    group_val: list[int] = []

    cursor = 0
    for gi, gsize in enumerate(group):
        cand_slice = slice(cursor, cursor + gsize)
        if gi in val_indices:
            X_val.extend(X[cand_slice])
            y_val.extend(y[cand_slice])
            group_val.append(gsize)
        else:
            X_train.extend(X[cand_slice])
            y_train.extend(y[cand_slice])
            group_train.append(gsize)
        cursor += gsize

    return {
        "X_train": X_train,
        "y_train": y_train,
        "group_train": group_train,
        "X_val": X_val,
        "y_val": y_val,
        "group_val": group_val,
    }


def train_lambdarank(
    X: list[list[float]],
    y: list[int],
    group: list[int],
    *,
    feature_names: list[str],
    output_path: Path | None = None,
    val_fraction: float = 0.2,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 30,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
) -> dict[str, Any]:
    """Entrena un LGBMRanker lambdarank y persiste el modelo.

    Args:
        X, y, group: training data según contrato de feedback_to_training_data.
        feature_names: lista de nombres de features (debe coincidir en orden
            con las columnas de X). Persistida con el modelo para audit.
        output_path: dónde guardar el .lgbm (default ~/.local/share/obsidian-rag/ranker.lgbm).
        val_fraction: fracción de groups para validation. 0 = sin val.
        num_boost_round: árboles máximo. Lightgbm puede early-stop antes.
        early_stopping_rounds: si val_fraction>0, para si val no mejora.
        learning_rate: shrinkage, 0.05 es conservador (más árboles).
        num_leaves: complejidad por árbol. 31 default — captura interacciones
            sin overfit en datasets chicos.
        seed: para shuffle del split.

    Returns:
        dict con paths del modelo + métricas + metadata.
    """
    import lightgbm as lgb
    import numpy as np

    output_path = output_path or _default_model_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not X or not y or not group:
        raise ValueError(
            "Training data está vacío. ¿Tenés feedback con paths_json y "
            "corrective_paths? Corré `rag feedback status` para ver."
        )

    if len(X) != len(y):
        raise ValueError(f"X y y tienen len distintos: {len(X)} vs {len(y)}")

    if sum(group) != len(X):
        raise ValueError(
            f"sum(group)={sum(group)} != len(X)={len(X)} — group sizes "
            f"no coinciden con candidates."
        )

    split = _validation_split(X, y, group, val_fraction=val_fraction, seed=seed)

    train_X = np.array(split["X_train"], dtype=np.float64)
    train_y = np.array(split["y_train"], dtype=np.int32)
    train_group = np.array(split["group_train"], dtype=np.int32)

    train_set = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        feature_name=feature_names,
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if split["X_val"]:
        val_X = np.array(split["X_val"], dtype=np.float64)
        val_y = np.array(split["y_val"], dtype=np.int32)
        val_group = np.array(split["group_val"], dtype=np.int32)
        val_set = lgb.Dataset(
            val_X, label=val_y, group=val_group,
            feature_name=feature_names, reference=train_set,
        )
        valid_sets.append(val_set)
        valid_names.append("val")

    params: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        # NDCG at 1, 3, 5 — match al ranker linear actual y al eval.
        "ndcg_eval_at": [1, 3, 5],
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        # Conservative regularization — datasets chicos overfit fácil.
        "min_data_in_leaf": max(5, len(train_y) // 50),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        # macOS ARM workaround: lightgbm con OpenMP hace segfault
        # intermitente cuando varios threads acceden al booster en
        # paralelo bajo dataset chico (verificado 2026-04-26 en M-series
        # con 73 queries / 1095 candidates). Forzar single-thread elimina
        # el segfault y la perf degradación es despreciable a este scale.
        "num_threads": 1,
    }

    callbacks = [lgb.log_evaluation(period=0)]  # silent
    if val_fraction > 0 and split["X_val"]:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        )

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    # Persist model (binary lightgbm format) + sidecar metadata.
    booster.save_model(str(output_path))
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train_queries": len(train_group),
        "n_train_candidates": len(train_y),
        "n_val_queries": len(split["group_val"]),
        "n_val_candidates": len(split["y_val"]),
        "params": {k: v for k, v in params.items() if not callable(v)},
        "best_iteration": booster.best_iteration if booster.best_iteration > 0 else None,
        "feature_importance_gain": dict(zip(
            feature_names,
            [float(v) for v in booster.feature_importance(importance_type="gain")],
        )),
        "feature_importance_split": dict(zip(
            feature_names,
            [float(v) for v in booster.feature_importance(importance_type="split")],
        )),
    }
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(output_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "best_iteration": metadata["best_iteration"],
        "n_train_queries": metadata["n_train_queries"],
        "n_val_queries": metadata["n_val_queries"],
    }


def persist_train_run_to_telemetry(
    conn: sqlite3.Connection,
    *,
    metadata: dict[str, Any],
    delta_vs_linear: float | None = None,
) -> None:
    """Inserta un row en rag_tune con la metadata del training run.

    Permite trackear evolución de modelos lambdarank a lo largo del tiempo
    igual que con `rag tune` linear. El campo `delta` queda NULL si no
    se hizo eval comparativo (se completa después con el output de
    `eval_lambdarank_vs_linear`).
    """
    try:
        conn.execute(
            """
            INSERT INTO rag_tune
              (ts, cmd, samples, seed, n_cases, baseline_json, best_json, delta)
            VALUES (?, 'tune-lambdarank', 1, 42, ?, ?, ?, ?)
            """,
            (
                metadata["trained_at"],
                metadata["n_train_queries"],
                json.dumps({"model": "linear", "ranker_json_active": True}),
                json.dumps({
                    "model": "lambdarank",
                    "n_features": metadata["n_features"],
                    "best_iteration": metadata["best_iteration"],
                    "feature_importance_gain": metadata["feature_importance_gain"],
                }),
                delta_vs_linear,
            ),
        )
    except sqlite3.OperationalError as exc:
        logger.warning(
            "Could not persist tune-lambdarank to rag_tune (table missing or "
            "schema mismatch): %s", exc,
        )
