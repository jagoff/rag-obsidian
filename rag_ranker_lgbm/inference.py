"""Inference: scorea candidatos con un modelo LightGBM lambdarank entrenado.

Diseño: API mínima compatible con cómo `apply_weighted_scores` del ranker
linear scorea — para que la integración a `rag/__init__.py` sea un swap
condicional sin tocar el resto de retrieve.

Patrón típico:
    from rag_ranker_lgbm import LambdaRankerScorer

    scorer = LambdaRankerScorer.load_default()  # idempotent caching
    scores = scorer.predict(candidates)         # list[float]
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

`load_default()` vive como classmethod con caching estático para que
varios callers lo invoquen sin pagar el cost de re-cargar el modelo
(el .lgbm pesa <1MB pero parsearlo lleva ~50ms).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache the loaded scorer per model_path. Process-local — el web server
# y el CLI son procesos distintos, cada uno carga independiente.
_scorer_cache: dict[str, "LambdaRankerScorer"] = {}
_cache_lock = threading.Lock()


class LambdaRankerScorer:
    """Wrapper sobre lgb.Booster con API mínima `predict(features)`.

    Construir directo NO suele ser lo que querés — usá `load_default()`
    para reusar el cached booster en múltiples llamadas dentro del proceso.
    """

    def __init__(self, booster, feature_names: list[str]):
        self._booster = booster
        self._feature_names = feature_names

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def predict(self, candidates: list[dict[str, Any]]) -> list[float]:
        """Scorea cada candidato y devuelve un score raw (mayor = más relevante).

        Args:
            candidates: lista de dicts con las mismas keys que devuelve
                `collect_ranker_features` en rag/__init__.py.

        Returns:
            list[float] con un score por candidato. NO está normalizado
            (lambdarank scores son valores raw que solo se usan para sort).
            El caller puede compararlos entre sí pero no across queries.
        """
        if not candidates:
            return []

        from rag_ranker_lgbm.features import _candidate_to_feature_vector

        # `has_recency_cue` es global a la query — todos los candidatos
        # comparten la misma. Tomamos el del primero.
        has_recency_cue = bool(candidates[0].get("has_recency_cue", False))

        X = [
            _candidate_to_feature_vector(cand, has_recency_cue)
            for cand in candidates
        ]
        # lgb.Booster.predict acepta list of lists o np.ndarray.
        try:
            import numpy as np
            scores = self._booster.predict(np.array(X, dtype=np.float64))
        except ImportError:
            # Sin numpy disponible (caso edge), fallback a list.
            scores = self._booster.predict(X)
        return [float(s) for s in scores]

    @classmethod
    def load(cls, model_path: str | Path) -> "LambdaRankerScorer":
        """Carga el booster desde disk sin caching (uso explícito)."""
        import lightgbm as lgb

        path_str = str(model_path)
        booster = lgb.Booster(model_file=path_str)
        # Recover feature names — booster.feature_name() retorna None si
        # se entrenó sin nombres explícitos. Fallback a las constantes.
        feat_names = booster.feature_name()
        if not feat_names:
            from rag_ranker_lgbm.features import FEATURE_NAMES
            feat_names = list(FEATURE_NAMES)
        return cls(booster, feat_names)

    @classmethod
    def load_default(
        cls, model_path: str | Path | None = None
    ) -> "LambdaRankerScorer | None":
        """Carga + cachea el modelo desde la default location.

        Returns None si el modelo no existe en disk (no error — el caller
        decide si hacer fallback al ranker linear).
        """
        from rag_ranker_lgbm.train import DEFAULT_MODEL_PATH

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            return None
        cache_key = str(path)
        with _cache_lock:
            scorer = _scorer_cache.get(cache_key)
            if scorer is not None:
                return scorer
            scorer = cls.load(path)
            _scorer_cache[cache_key] = scorer
            return scorer

    @classmethod
    def clear_cache(cls) -> None:
        """Util para tests / hot-reload tras retrain."""
        with _cache_lock:
            _scorer_cache.clear()
