"""LightGBM lambdarank — modelo ranker no-lineal para reemplazar la
combinación lineal de 11 pesos del `ranker.json` actual.

El ranker linear actual (`apply_weighted_scores` en `rag/__init__.py`) es:
  score = w1 * rerank + w2 * recency + ... + w11 * dwell_score

Limitación: no captura interacciones entre features. Por ejemplo, un chunk
con `click_prior=0.9` Y `recency_cue=high` en una query con `has_recency_cue=true`
debería rankear ALTÍSIMO — más alto que la suma lineal de los 3 features
sueltos. El modelo lineal no puede expresar eso. LightGBM lambdarank con
`num_leaves=31` aprende esos cruces automáticamente desde feedback.

Diseño Sprint 2 (commit asociado):

- `features.py`: feedback_to_training_data() — lee rag_feedback, replay
  collect_ranker_features() para reconstruir el feature vector de cada
  candidato visto por el user, y deriva labels desde corrective_path /
  paths_json / rating.
- `train.py`: train_lambdarank() — wrapper sobre lgb.LGBMRanker con
  objective='lambdarank', ndcg_eval_at=[1,3,5]. Persiste modelo a
  `~/.local/share/obsidian-rag/ranker.lgbm` + métricas a `rag_tune`.
- `inference.py`: LambdaRankerScorer — class con `predict(features) -> scores`.
  Usable desde `apply_weighted_scores` para replazar el dot product.
- `eval.py`: eval_lambdarank_vs_linear() — A/B sobre queries.yaml,
  reporta delta hit@5 / MRR / recall@5.

NO hay integración con `rag.py` en este commit — el modelo se entrena y
se evalúa OFFLINE. Si los números son mejores, follow-up commit hace el
switch via `RAG_RANKER_LAMBDARANK=1` env. Decisión conservadora: tocar el
core de retrieve es alto riesgo, queremos validar el modelo antes.

Próximos pasos esperados (no en este commit):
- Hard negative mining para enriquecer training data más allá del feedback
  histórico (que es ralo).
- Synthetic query generation desde el corpus para domain adaptation.
- A/B en producción con env var + auto-rollback si winrate cae.
"""

from rag_ranker_lgbm.features import (
    FEATURE_NAMES,
    feedback_to_training_data,
)
from rag_ranker_lgbm.hard_negatives import (
    get_negatives_stats,
    mine_hard_negatives_for_synthetic,
)
from rag_ranker_lgbm.inference import LambdaRankerScorer
from rag_ranker_lgbm.synthetic_queries import (
    CROSS_SOURCE_SOURCES,
    generate_synthetic_queries,
    generate_synthetic_queries_for_cross_source,
    get_synthetic_stats,
)
from rag_ranker_lgbm.train import (
    DEFAULT_MODEL_PATH,
    train_lambdarank,
)

__all__ = [
    "CROSS_SOURCE_SOURCES",
    "DEFAULT_MODEL_PATH",
    "FEATURE_NAMES",
    "LambdaRankerScorer",
    "feedback_to_training_data",
    "generate_synthetic_queries",
    "generate_synthetic_queries_for_cross_source",
    "get_negatives_stats",
    "get_synthetic_stats",
    "mine_hard_negatives_for_synthetic",
    "train_lambdarank",
]
