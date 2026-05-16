"""sqlite-vec collection adapter used by the public ``rag`` facade."""
from __future__ import annotations

import json
from pathlib import Path

import sqlite_vec

__all__ = [
    "_KNOWN_META_COLS",
    "_sanitize_table_suffix",
    "SqliteVecCollection",
    "SqliteVecClient",
]

_KNOWN_META_COLS = (
    # Note-collection metadata
    "file", "folder", "note", "tags", "hash", "outlinks",
    "created_ts", "parent", "title", "area", "type", "archived_at",
    "archived_from", "archived_reason", "contradicts", "ambient",
    # URL sub-index metadata
    "url", "anchor", "line", "source", "profile", "bookmark_folder",
)


def _sanitize_table_suffix(name: str) -> str:
    """Collection name → SQL-safe table suffix. obsidian_notes_v9_abc123 is
    already alphanumeric+underscore; this is defensive."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class SqliteVecCollection:
    """sqlite-vec-backed collection.

    Exposes the subset of the API used by rag.py:
      count(), get(), add(), delete(), query(), .id, .name
    """

    __slots__ = ("_db", "name", "_dim", "_vec", "_meta")

    def __init__(self, db: "sqlite3.Connection", name: str, dim: int = 1024):
        self._db = db
        self.name = name
        self._dim = dim
        suffix = _sanitize_table_suffix(name)
        self._vec = f"vec_{suffix}"
        self._meta = f"meta_{suffix}"
        self._ensure_tables()

    def _ensure_tables(self):
        # Meta table is dim-independent → always create.
        # vec0 table requires a known dim; defer to _ensure_vec_table(dim)
        # called from the first add() — allows tests with 4-dim mocks and
        # production with 1024-dim bge-m3 to coexist without hardcoding.
        int_cols = {"line"}
        real_cols = {"created_ts"}
        cols_sql = ", ".join(
            f"{c} INTEGER" if c in int_cols
            else f"{c} REAL" if c in real_cols
            else f"{c} TEXT"
            for c in _KNOWN_META_COLS if c != "created_ts"
        )
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self._meta} ("
            " rowid INTEGER PRIMARY KEY, "
            " chunk_id TEXT UNIQUE NOT NULL, "
            " document TEXT, "
            f"{cols_sql}, "
            " created_ts REAL, "
            " extra_json TEXT"
            ")"
        )
        self._db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._meta}_file "
            f"ON {self._meta}(file)"
        )
        self._db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._meta}_folder "
            f"ON {self._meta}(folder)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS rag_schema_version ("
            " table_name TEXT PRIMARY KEY, "
            " version INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._db.commit()

    def _ensure_vec_table(self, dim: int):
        """Create vec0 table lazily on first insert with the actual dim.
        If the table already exists with a different dim, drop+recreate
        (destructive — caller must own the decision, e.g. `index --reset`
        or switching between test/prod)."""
        existing = self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (self._vec,),
        ).fetchone()
        if existing:
            sql = existing[0]
            if f"float[{dim}]" in sql:
                return
            # Dim mismatch → recreate. Only safe when no rows yet.
            row_ct = self._db.execute(
                f"SELECT COUNT(*) FROM {self._meta}"
            ).fetchone()[0]
            if row_ct > 0:
                raise RuntimeError(
                    f"vec0 dim mismatch in '{self._vec}': existing != {dim} "
                    f"and {row_ct} meta rows exist. Drop/recreate needed."
                )
            self._db.execute(f"DROP TABLE {self._vec}")
        self._db.execute(
            f"CREATE VIRTUAL TABLE {self._vec} "
            f"USING vec0(embedding float[{dim}])"
        )
        self._dim = dim

    @property
    def id(self) -> str:
        """Monotonic schema version bumped on every destructive write.
        Used by _load_corpus() to detect stale BM25 cache — a new `id`
        signals the collection was destructively rewritten.

        Mismo guard defensivo que `count()`: bajo race / concurrent shared
        connection, `execute` puede tirar `InterfaceError: bad parameter
        or other API misuse` (silent_errors.jsonl 2026-05-01) o la tabla
        puede no existir todavía. Devolvemos un fallback determinístico
        (`<name>:0`) en vez de propagar — `_load_corpus` lo trata como
        cache hit del primer rebuild, peor caso un rebuild extra.
        """
        import sqlite3 as _sqlite3
        try:
            row = self._db.execute(
                "SELECT version FROM rag_schema_version WHERE table_name = ?",
                (self.name,),
            ).fetchone()
        except (_sqlite3.InterfaceError, _sqlite3.OperationalError):
            return f"{self.name}:0"
        try:
            v = row[0] if row else 0
        except (IndexError, TypeError):
            v = 0
        return f"{self.name}:{v}"

    def _bump_version(self):
        self._db.execute(
            "INSERT INTO rag_schema_version(table_name, version) VALUES(?, 1) "
            "ON CONFLICT(table_name) DO UPDATE SET version = version + 1",
            (self.name,),
        )

    def count(self) -> int:
        # Defensive: bajo race con `index --reset` / `delete_collection` la
        # `meta_*` table puede estar mid-recreate; `fetchone()` puede devolver
        # None o `()`. Y bajo concurrent thread access al shared sqlite3
        # connection (`check_same_thread=False`), el statement state puede
        # corromperse → `InterfaceError: bad parameter or other API misuse`.
        # En ambos casos devolvemos 0 — el caller (`_load_corpus`) interpreta
        # un count delta como invalidación de cache, lo peor que pasa es un
        # rebuild extra en el siguiente call. Mejor que tirar la query entera
        # vía graph_expand.outer (silent_errors.jsonl 2026-05-01: 23 hits con
        # tracebacks pegando acá).
        import sqlite3 as _sqlite3
        try:
            row = self._db.execute(
                f"SELECT COUNT(*) FROM {self._meta}"
            ).fetchone()
        except (_sqlite3.InterfaceError, _sqlite3.OperationalError):
            # InterfaceError → connection state corrupto; OperationalError →
            # tabla no existe (drop/create race). Ambos: degrade graceful.
            return 0
        if not row:
            return 0
        try:
            return int(row[0])
        except (IndexError, TypeError, ValueError):
            return 0

    @staticmethod
    def _build_where(where: dict | None) -> tuple[str, list]:
        """Translate a dict-based `where` filter to a SQL WHERE clause.

        Supported:
          {field: value}              → `field = ?`
          {field: {"$in": [...]}}     → `field IN (?, ?, …)`
          {field: {"$contains": "x"}} → `field LIKE '%x%'`
          {field: {"$gte": n}}        → `field >= ?`  (also $gt/$lte/$lt/$ne)
          {"$and": [cond1, cond2]}    → `(cond1) AND (cond2)`
          {"$or": [cond1, cond2]}     → `(cond1) OR (cond2)`
        """
        def _expand(cond: dict) -> tuple[str, list]:
            if not cond:
                return "1 = 1", []
            parts: list[str] = []
            params: list = []
            for k, v in cond.items():
                if k == "$and":
                    subs = [_expand(c) for c in v]
                    parts.append(" AND ".join(f"({s[0]})" for s in subs))
                    for s in subs:
                        params.extend(s[1])
                elif k == "$or":
                    subs = [_expand(c) for c in v]
                    parts.append(" OR ".join(f"({s[0]})" for s in subs))
                    for s in subs:
                        params.extend(s[1])
                elif isinstance(v, dict):
                    # Field-level operator dict
                    field_parts = []
                    for op, val in v.items():
                        if op == "$in":
                            vals = list(val)
                            if not vals:
                                field_parts.append("0 = 1")
                            else:
                                ph = ",".join("?" * len(vals))
                                field_parts.append(f"{k} IN ({ph})")
                                params.extend(vals)
                        elif op == "$nin":
                            vals = list(val)
                            if not vals:
                                field_parts.append("1 = 1")
                            else:
                                ph = ",".join("?" * len(vals))
                                field_parts.append(f"({k} NOT IN ({ph}) OR {k} IS NULL)")
                                params.extend(vals)
                        elif op == "$contains":
                            field_parts.append(f"{k} LIKE ?")
                            params.append(f"%{val}%")
                        elif op == "$gte":
                            field_parts.append(f"{k} >= ?")
                            params.append(val)
                        elif op == "$lte":
                            field_parts.append(f"{k} <= ?")
                            params.append(val)
                        elif op == "$gt":
                            field_parts.append(f"{k} > ?")
                            params.append(val)
                        elif op == "$lt":
                            field_parts.append(f"{k} < ?")
                            params.append(val)
                        elif op == "$ne":
                            field_parts.append(f"({k} != ? OR {k} IS NULL)")
                            params.append(val)
                        elif op == "$eq":
                            field_parts.append(f"{k} = ?")
                            params.append(val)
                        else:
                            # Unknown operator — fall through to equality of dict
                            # serialised as JSON. Defensive; shouldn't happen.
                            field_parts.append(f"{k} = ?")
                            params.append(json.dumps(v))
                    if field_parts:
                        parts.append(" AND ".join(field_parts))
                else:
                    parts.append(f"{k} = ?")
                    params.append(v)
            return " AND ".join(parts), params

        if not where:
            return "", []
        sql, params = _expand(where)
        return " WHERE " + sql, params

    def _row_to_meta(self, row, meta_col_names):
        """Reconstruct metadata dict from meta_* row."""
        out = {}
        for i, col in enumerate(meta_col_names):
            v = row[i]
            if v is None:
                continue
            out[col] = v
        # Merge extra_json if present
        extra = out.pop("extra_json", None)
        if extra:
            try:
                out.update(json.loads(extra))
            except (json.JSONDecodeError, TypeError):
                pass
        return out

    def get(self, ids=None, where=None, include=None) -> dict:
        include = include or []
        want_docs = "documents" in include
        want_metas = "metadatas" in include
        want_embs = "embeddings" in include

        meta_cols = list(_KNOWN_META_COLS) + ["extra_json"]
        select_cols = ["rowid", "chunk_id"]
        if want_docs:
            select_cols.append("document")
        if want_metas:
            select_cols.extend(meta_cols)

        sql = f"SELECT {', '.join(select_cols)} FROM {self._meta}"
        params: list = []
        if ids is not None:
            ids_list = list(ids)
            if not ids_list:
                base = {"ids": []}
                if want_docs:
                    base["documents"] = []
                if want_metas:
                    base["metadatas"] = []
                if want_embs:
                    base["embeddings"] = []
                return base
            placeholders = ",".join("?" * len(ids_list))
            sql += f" WHERE chunk_id IN ({placeholders})"
            params = ids_list
        elif where:
            w_sql, w_params = self._build_where(where)
            sql += w_sql
            params = w_params

        rows = self._db.execute(sql, params).fetchall()
        result_ids = [r[1] for r in rows]
        out: dict = {"ids": result_ids}

        if want_docs:
            doc_idx = 2
            out["documents"] = [r[doc_idx] for r in rows]
        if want_metas:
            meta_start = 2 + (1 if want_docs else 0)
            out["metadatas"] = [self._row_to_meta(r[meta_start:], meta_cols) for r in rows]
        if want_embs:
            import numpy as _np
            rowids = [r[0] for r in rows]
            embed_map: dict[int, list[float]] = {}
            vec_exists = self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                (self._vec,),
            ).fetchone() is not None
            if rowids and vec_exists:
                placeholders = ",".join("?" * len(rowids))
                for rid, raw in self._db.execute(
                    f"SELECT rowid, embedding FROM {self._vec} WHERE rowid IN ({placeholders})",
                    rowids,
                ).fetchall():
                    arr = _np.frombuffer(raw, dtype="float32")
                    embed_map[rid] = arr.tolist()
            out["embeddings"] = [embed_map.get(r[0]) for r in rows]

        return out

    def add(self, ids, embeddings, documents, metadatas):
        known = _KNOWN_META_COLS
        # Lazily create the vec0 table with the actual embedding dim.
        embeddings = [list(e) for e in embeddings]
        if embeddings:
            self._ensure_vec_table(len(embeddings[0]))

        # Pre-compute cols_list and SQL template once (not per-row).
        cols_list = ["chunk_id", "document"] + list(known) + ["extra_json"]
        placeholders = ",".join("?" * len(cols_list))
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols_list if c != "chunk_id")
        # RETURNING rowid: collapse INSERT + SELECT rowid into a single query.
        # Requires SQLite ≥3.35 (available since CPython 3.10 ships 3.35+;
        # our runtime is 3.53). Works for both INSERT and ON CONFLICT upsert paths.
        meta_sql = (
            f"INSERT INTO {self._meta}({','.join(cols_list)}) VALUES({placeholders}) "
            f"ON CONFLICT(chunk_id) DO UPDATE SET {updates} RETURNING rowid"
        )
        vec_delete_sql = f"DELETE FROM {self._vec} WHERE rowid = ?"
        vec_insert_sql = f"INSERT INTO {self._vec}(rowid, embedding) VALUES(?, ?)"

        with self._db:
            for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                extra = {k: v for k, v in meta.items() if k not in known}
                extra_json = json.dumps(extra, ensure_ascii=False) if extra else None
                created_ts = meta.get("created_ts")
                try:
                    created_ts = float(created_ts) if created_ts is not None else None
                except (TypeError, ValueError):
                    created_ts = None
                values = [cid, doc]
                for col in known:
                    if col == "created_ts":
                        values.append(created_ts)
                    else:
                        v = meta.get(col)
                        if v is None or isinstance(v, (int, float, bool, str)):
                            values.append(v)
                        else:
                            values.append(str(v))
                values.append(extra_json)
                rowid = self._db.execute(meta_sql, values).fetchone()[0]
                # Replace vec entry (vec0 doesn't support upsert directly)
                self._db.execute(vec_delete_sql, (rowid,))
                self._db.execute(vec_insert_sql,
                                 (rowid, sqlite_vec.serialize_float32(list(emb))))
            self._bump_version()

    def update(self, ids, metadatas):
        """Metadata-only update — keeps the embedding + document untouched.

        Chromadb-era callers (`_maybe_backfill_created_ts` en rag.py:18434)
        esperan `col.update(ids=..., metadatas=...)`. Sin este método el
        backfill lazy fallaba con
        `AttributeError: 'SqliteVecCollection' object has no attribute 'update'`
        y cada query con `date_range` omitía silenciosamente los chunks
        pre-temporal (verificado en `web.log` 2026-04-20).

        Updates only the columns present in each meta dict. Missing columns
        stay at their current value (no NULL clobbering). `extra_json` is
        fully rewritten from the non-known-cols present in the incoming
        meta — callers should pass the FULL merged metadata dict (same
        convention as `add`), not a partial patch.
        """
        if not ids:
            return
        known = _KNOWN_META_COLS
        with self._db:
            for cid, meta in zip(ids, metadatas):
                if meta is None:
                    continue
                # Split into first-class cols vs the extra bag.
                set_clauses: list[str] = []
                params: list = []
                for col in known:
                    if col not in meta:
                        continue
                    v = meta.get(col)
                    if col == "created_ts":
                        try:
                            v = float(v) if v is not None else None
                        except (TypeError, ValueError):
                            v = None
                    elif v is not None and not isinstance(v, (int, float, bool, str)):
                        v = str(v)
                    set_clauses.append(f"{col}=?")
                    params.append(v)
                # Rewrite `extra_json` from every non-known key present in
                # the payload — mirrors `add()`'s normalization so a roundtrip
                # add→update stays shape-stable.
                extra_keys = [k for k in meta.keys() if k not in known]
                if extra_keys:
                    extra = {k: meta[k] for k in extra_keys}
                    set_clauses.append("extra_json=?")
                    params.append(json.dumps(extra, ensure_ascii=False) if extra else None)
                if not set_clauses:
                    continue
                params.append(cid)
                self._db.execute(
                    f"UPDATE {self._meta} SET {', '.join(set_clauses)} WHERE chunk_id=?",
                    params,
                )
            self._bump_version()

    def delete(self, ids=None, where=None):
        rowids: list[int] = []
        if ids is not None:
            if not ids:
                return
            placeholders = ",".join("?" * len(ids))
            rowids = [r[0] for r in self._db.execute(
                f"SELECT rowid FROM {self._meta} WHERE chunk_id IN ({placeholders})",
                list(ids),
            ).fetchall()]
        elif where:
            w_sql, w_params = self._build_where(where)
            rowids = [r[0] for r in self._db.execute(
                f"SELECT rowid FROM {self._meta}" + w_sql, w_params,
            ).fetchall()]
        if not rowids:
            return
        with self._db:
            placeholders = ",".join("?" * len(rowids))
            self._db.execute(
                f"DELETE FROM {self._vec} WHERE rowid IN ({placeholders})", rowids,
            )
            self._db.execute(
                f"DELETE FROM {self._meta} WHERE rowid IN ({placeholders})", rowids,
            )
            self._bump_version()

    def query(self, query_embeddings, n_results: int = 10,
              where=None, include=None) -> dict:
        """Semantic search. Returns batched results: {ids, documents,
        metadatas, distances} each as list[list[...]] — one sublist per
        query embedding."""
        include = include or ["documents", "metadatas", "distances"]
        want_docs = "documents" in include
        want_metas = "metadatas" in include
        want_dist = "distances" in include

        meta_cols = list(_KNOWN_META_COLS) + ["extra_json"]
        select_cols = ["m.chunk_id"]
        if want_docs:
            select_cols.append("m.document")
        if want_metas:
            select_cols.extend(f"m.{c}" for c in meta_cols)
        select_cols.append("v.distance")
        # Trailing v.rowid → used only for Python-side stable tie-breaking.
        select_cols.append("v.rowid")

        where_sql, where_params = self._build_where(where)
        # vec0's MATCH is restrictive — can't chain arbitrary predicates in
        # the same SELECT. Split the knn into a CTE and apply metadata
        # filters on the outer SELECT. Over-fetch by 4× to compensate for
        # post-filtering losing top candidates.
        fetch_n = int(n_results)
        post_filter = bool(where_sql)
        if post_filter:
            fetch_n = int(n_results) * 4 + 20

        out: dict[str, list] = {"ids": []}
        if want_docs:
            out["documents"] = []
        if want_metas:
            out["metadatas"] = []
        if want_dist:
            out["distances"] = []

        vec_exists = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (self._vec,),
        ).fetchone() is not None
        if not vec_exists:
            for _ in query_embeddings:
                out["ids"].append([])
                if want_docs:
                    out["documents"].append([])
                if want_metas:
                    out["metadatas"].append([])
                if want_dist:
                    out["distances"].append([])
            return out

        for qemb in query_embeddings:
            # Step 1: vec0 knn into a subquery (no extra predicates allowed).
            # Step 2: outer SELECT joins meta, applies user `where`, limits to n_results.
            outer_where = where_sql  # already starts with " WHERE " if non-empty
            # Secondary sort by rowid keeps tied distances in insertion order
            # — without it, vec0 may return ties non-deterministically.
            sql = (
                f"SELECT {', '.join(select_cols)} FROM ("
                f"  SELECT rowid, distance FROM {self._vec} "
                f"  WHERE embedding MATCH ? AND k = ? ORDER BY distance"
                f") v JOIN {self._meta} m ON v.rowid = m.rowid"
                f"{outer_where} "
                f"LIMIT ?"
            )
            params = (
                [sqlite_vec.serialize_float32(list(qemb)), fetch_n]
                + where_params
                + [int(n_results)]
            )
            rows = self._db.execute(sql, params).fetchall()
            # Python-side stable sort by (distance, rowid) — sqlite-vec ties
            # resolve arbitrarily inside vec0, so we enforce insertion-order
            # for ties here. Columns layout: chunk_id [doc] [meta*] distance rowid.
            rows = sorted(rows, key=lambda r: (r[-2], r[-1]))
            ids = [r[0] for r in rows]
            out["ids"].append(ids)
            if want_docs:
                out["documents"].append([r[1] for r in rows])
            if want_metas:
                ms_start = 2 if want_docs else 1
                out["metadatas"].append(
                    [self._row_to_meta(r[ms_start:ms_start + len(meta_cols)], meta_cols)
                     for r in rows]
                )
            if want_dist:
                out["distances"].append([r[-2] for r in rows])

        return out


class SqliteVecClient:
    """sqlite-vec client. All collections share one SQLite file
    (ragvec.db inside path).

    Concurrency contract — READ THIS BEFORE ADDING PARALLELISM:

    - The connection uses `check_same_thread=False` + `isolation_level=None`
      (autocommit) + WAL + NORMAL synchronous. Multiple threads can issue
      statements on the same connection; SQLite serialises them internally.
    - Readers are **eventually consistent**, not snapshot-isolated. A query
      that runs mid-write may see the post-write state for some rows and the
      pre-write state for others (WAL gives durability, not MVCC snapshots
      at the Python API level since we autocommit). Good enough for the
      retrieval pipeline, which tolerates a chunk appearing/disappearing
      between calls.
    - Writes are serialised **at the application layer** by
      `_collection_write_lock()` — the module-wide file lock under
      `~/.local/share/obsidian-rag/.collection-ops.lock`. DO NOT rely on
      SQLite's busy_timeout alone for correctness; the write lock also
      guards the JSONL audit trail + schema-version bump.
    - Telemetry writes inside `RAG_STATE_SQL` tables use explicit
      `BEGIN IMMEDIATE` + `COMMIT` wrappers (see `_sql_upsert`) because
      they batch multiple rows. Vec writes (`col.add`) are single-row and
      run under autocommit.

    If you add snapshot reads (e.g. "give me a consistent view during a
    long analytical scan"), wrap the read in `BEGIN DEFERRED` / `COMMIT`
    explicitly — autocommit + WAL is not enough.
    """

    def __init__(self, path):
        import sqlite3 as _sqlite3
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._file = self.path / "ragvec.db"
        self._db = _sqlite3.connect(str(self._file), check_same_thread=False,
                                    isolation_level=None)
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        # busy_timeout 60s (era 10s pre-2026-04-30): bajo bursts concurrentes
        # de ingesters (calls + safari + drive disparados al mismo tiempo
        # tras `rag setup` o recovery), 10s no bastaba — los procesos hijos
        # tiraban `database is locked` y exit=1, dejando el plist en estado
        # error hasta el próximo schedule. 60s es el mismo orden que
        # telemetry.db y suficiente para cualquier contención normal.
        self._db.execute("PRAGMA busy_timeout=60000")
        # cache_size + mmap_size — audit 2026-05-04: ragvec.db default 8 MB
        # cache (cache_size=2000 pages × 4096 bytes) era ridículo para una
        # DB de ~250 MB; bajos hit rates en page cache + cero memory-mapping
        # multiplicaban IO. 64 MB de cache cubre todo el footprint vivo
        # típico (~50-100 MB) y el mmap de 256 MB le permite al kernel
        # servir lecturas hot directamente desde page cache sin syscalls.
        # Negativo = KB. Idempotente — PRAGMA noop si ya está seteado.
        self._db.execute("PRAGMA cache_size=-65536")
        self._db.execute("PRAGMA mmap_size=268435456")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS rag_schema_version ("
            " table_name TEXT PRIMARY KEY, "
            " version INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._db.commit()
        self._collections: dict[str, SqliteVecCollection] = {}
        # Post 2026-04-21 split: telemetry tables live in telemetry.db (see
        # _TELEMETRY_DB_FILENAME + _ragvec_state_conn). SqliteVecClient no
        # longer creates rag_* tables on ragvec.db — that was a safety net
        # for a pre-T10 code path that's gone. RAG_STATE_SQL env is still
        # respected by plist trail-config but now a no-op here.

    def get_or_create_collection(self, name: str, metadata=None):
        # metadata arg ignored (was `{"hnsw:space": "cosine"}`); sqlite-vec
        # uses cosine distance natively via vec0 MATCH.
        if name not in self._collections:
            self._collections[name] = SqliteVecCollection(self._db, name)
        return self._collections[name]

    def list_collections(self):
        rows = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'meta_%' "
            "ORDER BY name"
        ).fetchall()
        # Strip "meta_" prefix → collection name
        names = [r[0][5:] for r in rows]
        # Return lightweight objects with .name (callers only use .name)
        class _ColRef:
            def __init__(self, n): self.name = n
        return [_ColRef(n) for n in names]

    def delete_collection(self, name: str):
        suffix = _sanitize_table_suffix(name)
        with self._db:
            self._db.execute(f"DROP TABLE IF EXISTS vec_{suffix}")
            self._db.execute(f"DROP TABLE IF EXISTS meta_{suffix}")
            self._db.execute(
                "DELETE FROM rag_schema_version WHERE table_name = ?", (name,)
            )
        self._collections.pop(name, None)

