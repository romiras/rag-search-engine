from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .store import Storage

SCHEMA_VERSION = 1
DEFAULT_CLUSTER_SEED = 7
K_MIN = 8
K_MAX = 24
K_START = 12
N_RESTARTS = 10
MAX_KMEANS_ITER = 50
MAX_SILHOUETTE_SAMPLE = 564
MAX_GLOBAL_HEADINGS = 80
MAX_DOC_TOP_HEADINGS = 5

BREADCRUMB_RE = re.compile(r"^\[(.+?)\] \| ")


def slugify_root(label: str) -> str:
    """Builds a stable lowercase slug from a root path label."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "root"


def parse_chunk_breadcrumb(
    content: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts (H1, H2, H3) from a chunk formatted as `[H1 > H2 > H3] | body`.
    Ignores empty segments and the `...` placeholder inserted by MarkdownChunker.
    Preserves raw heading strings verbatim.
    """
    match = BREADCRUMB_RE.match(content)
    if not match:
        return (None, None, None)

    parts = match.group(1).split(" > ")
    cleaned: List[Optional[str]] = [None, None, None]
    for idx in range(min(3, len(parts))):
        seg = parts[idx]
        if seg and seg.strip() and seg != "...":
            cleaned[idx] = seg
    return (cleaned[0], cleaned[1], cleaned[2])


def compute_source_fingerprint(rows: Sequence[Tuple[int, str, str, int]]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint over (path, content_hash, chunk_count)
    across all indexed documents.
    """
    hasher = hashlib.sha256()
    sorted_rows = sorted(rows, key=lambda r: r[1])
    for _doc_id, path, content_hash, chunk_count in sorted_rows:
        hasher.update(f"{path}\0{content_hash}\0{chunk_count}\n".encode("utf-8"))
    return hasher.hexdigest()


def _build_root_definitions(include_paths: Sequence[str]) -> List[Dict[str, str]]:
    """Creates root metadata entries with unique stable ids from configured include_paths."""
    roots: List[Dict[str, str]] = []
    used_ids: Dict[str, int] = {}

    for raw_path in include_paths:
        if not raw_path:
            continue
        norm_prefix = os.path.normpath(raw_path)
        label = Path(norm_prefix).name or norm_prefix
        base_id = slugify_root(label)
        if base_id == "other":
            base_id = "root-other"

        count = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = count
        root_id = base_id if count == 1 else f"{base_id}-{count}"

        roots.append(
            {
                "id": root_id,
                "label": label,
                "path_prefix": raw_path.rstrip("/") if raw_path != "/" else "/",
                "_norm_prefix": norm_prefix,
            }
        )
    return roots


def _match_root_id(doc_path: str, root_defs: Sequence[Dict[str, str]]) -> str:
    """Matches a document path to configured roots by longest path prefix."""
    norm_doc = os.path.normpath(doc_path)
    best_id = "other"
    best_len = -1

    for rdef in root_defs:
        prefix = rdef["_norm_prefix"]
        if norm_doc == prefix or norm_doc.startswith(prefix.rstrip(os.sep) + os.sep):
            if len(prefix) > best_len:
                best_len = len(prefix)
                best_id = rdef["id"]
    return best_id


def _kmeans_single_run(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = MAX_KMEANS_ITER,
) -> Tuple[np.ndarray, float]:
    """
    Runs one cosine k-means trial on L2-normalized rows X (N, D).
    Uses k-means++ initialization and guarantees non-empty clusters.
    Returns (labels, inertia).
    """
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features), dtype=np.float32)

    # k-means++ initialization using cosine distance d = 1 - cos_sim
    first_idx = int(rng.integers(0, n_samples))
    centroids[0] = X[first_idx]
    chosen_mask = np.zeros(n_samples, dtype=bool)
    chosen_mask[first_idx] = True

    min_dist = np.maximum(0.0, 1.0 - (X @ centroids[0]))
    min_dist[first_idx] = 0.0

    for c_idx in range(1, k):
        weights = min_dist**2
        weight_sum = float(np.sum(weights))
        if weight_sum > 1e-12:
            probs = weights / weight_sum
            next_idx = int(rng.choice(n_samples, p=probs))
        else:
            available = np.flatnonzero(~chosen_mask)
            next_idx = int(
                rng.choice(available)
                if len(available) > 0
                else rng.integers(0, n_samples)
            )
        centroids[c_idx] = X[next_idx]
        chosen_mask[next_idx] = True
        dist_to_new = np.maximum(0.0, 1.0 - (X @ centroids[c_idx]))
        min_dist = np.minimum(min_dist, dist_to_new)
        min_dist[chosen_mask] = 0.0

    prev_labels: Optional[np.ndarray] = None
    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        sims = X @ centroids.T
        labels = np.argmax(sims, axis=1).astype(np.int32)

        # Ensure no empty clusters
        counts = np.bincount(labels, minlength=k)
        empty_clusters = np.flatnonzero(counts == 0)
        if len(empty_clusters) > 0:
            point_best_sim = sims[np.arange(n_samples), labels]
            # Pick points farthest from their assigned centroid
            farthest_order = np.argsort(point_best_sim)
            donor_ptr = 0
            for empty_c in empty_clusters:
                while (
                    donor_ptr < n_samples
                    and counts[labels[farthest_order[donor_ptr]]] <= 1
                ):
                    donor_ptr += 1
                if donor_ptr >= n_samples:
                    break
                chosen_pt = int(farthest_order[donor_ptr])
                counts[labels[chosen_pt]] -= 1
                labels[chosen_pt] = empty_c
                counts[empty_c] += 1
                donor_ptr += 1

        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

        # Update centroids and L2-normalize
        for c_idx in range(k):
            member_mask = labels == c_idx
            if np.any(member_mask):
                mean_vec = np.mean(X[member_mask], axis=0, dtype=np.float64)
                norm = float(np.linalg.norm(mean_vec))
                if norm > 1e-12:
                    centroids[c_idx] = (mean_vec / norm).astype(np.float32)
                else:
                    centroids[c_idx] = mean_vec.astype(np.float32)

    assigned_sims = np.sum(X * centroids[labels], axis=1)
    inertia = float(np.sum(np.maximum(0.0, 1.0 - assigned_sims)))
    return labels, inertia


def _kmeans_cosine(
    X: np.ndarray,
    k: int,
    seed: int = DEFAULT_CLUSTER_SEED,
    n_restarts: int = N_RESTARTS,
) -> Tuple[np.ndarray, float]:
    """Runs `n_restarts` cosine k-means restarts for a fixed `k` and keeps the lowest inertia."""
    rng = np.random.default_rng(seed + k * 1009)
    best_labels: Optional[np.ndarray] = None
    best_inertia = float("inf")

    for _ in range(n_restarts):
        labels, inertia = _kmeans_single_run(X, k, rng)
        if inertia < best_inertia or best_labels is None:
            best_inertia = inertia
            best_labels = labels

    assert best_labels is not None
    return best_labels, best_inertia


def _mean_silhouette_from_dist_matrix(
    dist_matrix: np.ndarray, labels: np.ndarray
) -> float:
    """
    Computes exact mean silhouette score from a precomputed (S, S) distance matrix.
    Singleton clusters receive silhouette 0.0 per standard definition.
    """
    n = len(labels)
    if n < 2:
        return 0.0

    uniq_clusters, inverse_idx, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    n_clusters = len(uniq_clusters)
    if n_clusters < 2:
        return -1.0

    # cluster_dist_sums[i, c] = sum of distances from sample point i to all points in cluster c
    cluster_Membership = np.zeros((n, n_clusters), dtype=np.float64)
    cluster_Membership[np.arange(n), inverse_idx] = 1.0
    cluster_dist_sums = dist_matrix @ cluster_Membership  # shape (n, n_clusters)

    # Own-cluster mean distance a(i)
    own_sums = cluster_dist_sums[np.arange(n), inverse_idx]
    own_counts = counts[inverse_idx]
    a = np.where(own_counts > 1, own_sums / np.maximum(1, own_counts - 1), 0.0)

    # Nearest other-cluster mean distance b(i)
    mean_to_clusters = cluster_dist_sums / counts[None, :]
    mean_to_clusters[np.arange(n), inverse_idx] = np.inf
    b = np.min(mean_to_clusters, axis=1)

    denom = np.maximum(a, b)
    sil = np.where((own_counts > 1) & (denom > 1e-12), (b - a) / denom, 0.0)
    return float(np.mean(sil))


def cluster_documents(
    X: np.ndarray,
    seed: int = DEFAULT_CLUSTER_SEED,
) -> Tuple[np.ndarray, int]:
    """
    Clusters L2-normalized document matrix X (N, D) using cosine k-means.
    Selects k in 8..24 (starting at 12) to maximize mean silhouette on <=564 points.
    Gracefully handles small N (< 8) for unit tests and tiny indexes.
    Returns (labels, chosen_k).
    """
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=np.int32), 0
    if n_samples == 1:
        return np.zeros(1, dtype=np.int32), 1
    if n_samples == 2:
        labels, _ = _kmeans_cosine(X, 2, seed=seed)
        return labels, 2

    # Candidate k range
    if n_samples <= K_MIN:
        candidates = list(range(2, n_samples))
    elif n_samples == K_MIN + 1:
        candidates = [K_MIN]
    else:
        max_k = min(K_MAX, n_samples - 1)
        all_k = list(range(K_MIN, max_k + 1))
        if K_START in all_k:
            candidates = [K_START] + [k for k in all_k if k != K_START]
        else:
            candidates = all_k

    if len(candidates) == 1:
        chosen_k = candidates[0]
        labels, _ = _kmeans_cosine(X, chosen_k, seed=seed)
        return labels, chosen_k

    # Precompute sample cosine distance matrix once (at most 564 points)
    if n_samples > MAX_SILHOUETTE_SAMPLE:
        sample_rng = np.random.default_rng(seed)
        sample_idx = np.sort(
            sample_rng.choice(n_samples, size=MAX_SILHOUETTE_SAMPLE, replace=False)
        )
        X_sample = X[sample_idx]
    else:
        sample_idx = None
        X_sample = X

    sim_matrix = np.clip(X_sample @ X_sample.T, -1.0, 1.0)
    dist_matrix = np.maximum(0.0, 1.0 - sim_matrix).astype(np.float64)
    np.fill_diagonal(dist_matrix, 0.0)

    best_k = candidates[0]
    best_labels: Optional[np.ndarray] = None
    best_score = -float("inf")

    for k in candidates:
        labels, _ = _kmeans_cosine(X, k, seed=seed)
        sample_labels = labels[sample_idx] if sample_idx is not None else labels
        score = _mean_silhouette_from_dist_matrix(dist_matrix, sample_labels)
        if score > best_score or best_labels is None:
            best_score = score
            best_k = k
            best_labels = labels

    assert best_labels is not None
    return best_labels, best_k


def build_overview_payload(
    storage: Storage,
    include_paths: Sequence[str],
    model_name: str,
    seed: int = DEFAULT_CLUSTER_SEED,
) -> Tuple[Dict[str, Any], str]:
    """
    Builds the schema v1 overview JSON payload from SQLite storage without invoking Embedder.
    Returns (payload_dict, source_fingerprint).
    """
    fp_rows = storage.get_corpus_fingerprint_rows()
    source_fingerprint = compute_source_fingerprint(fp_rows)
    docs_raw, chunks_raw = storage.get_overview_source_data()

    root_defs = _build_root_definitions(include_paths)
    root_doc_counts: Dict[str, int] = {r["id"]: 0 for r in root_defs}
    root_chunk_counts: Dict[str, int] = {r["id"]: 0 for r in root_defs}
    root_doc_counts["other"] = 0
    root_chunk_counts["other"] = 0

    doc_root_map: Dict[int, str] = {}
    for doc_id, path, _content_hash in docs_raw:
        rid = _match_root_id(path, root_defs)
        doc_root_map[doc_id] = rid
        root_doc_counts[rid] = root_doc_counts.get(rid, 0) + 1

    # Per-document chunk & heading accumulators
    doc_chunk_counts: Dict[int, int] = defaultdict(int)
    doc_first_h1: Dict[int, str] = {}
    doc_heading_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    doc_heading_first_seen: Dict[int, Dict[str, int]] = defaultdict(dict)
    doc_h1_chunk_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    doc_h12_chunk_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    doc_vector_sums: Dict[int, np.ndarray] = {}
    doc_vector_counts: Dict[int, int] = defaultdict(int)

    # Global H1/H2 heading cloud accumulators
    global_heading_chunk_counts: Counter[str] = Counter()
    global_heading_docs: Dict[str, set[int]] = defaultdict(set)
    global_heading_level: Dict[str, int] = {}

    for _chunk_id, doc_id, content, emb_blob in chunks_raw:
        if doc_id not in doc_root_map:
            continue
        doc_chunk_counts[doc_id] += 1
        rid = doc_root_map[doc_id]
        root_chunk_counts[rid] = root_chunk_counts.get(rid, 0) + 1

        h1, h2, h3 = parse_chunk_breadcrumb(content)
        if h1 and doc_id not in doc_first_h1:
            doc_first_h1[doc_id] = h1

        # Per-document top_headings (H1, H2, and H3 if in top 5 for that file)
        seen_in_chunk_doc: set[str] = set()
        for h in (h1, h2, h3):
            if h and h not in seen_in_chunk_doc:
                seen_in_chunk_doc.add(h)
                doc_heading_counts[doc_id][h] += 1
                if h not in doc_heading_first_seen[doc_id]:
                    doc_heading_first_seen[doc_id][h] = len(
                        doc_heading_first_seen[doc_id]
                    )

        # Cluster label accumulators (H1 and H1/H2)
        if h1:
            doc_h1_chunk_counts[doc_id][h1] += 1
            doc_h12_chunk_counts[doc_id][h1] += 1
        if h2 and h2 != h1:
            doc_h12_chunk_counts[doc_id][h2] += 1

        # Global H1/H2 heading cloud
        for level, h in ((1, h1), (2, h2)):
            if not h:
                continue
            if level == 2 and h == h1:
                continue
            global_heading_chunk_counts[h] += 1
            global_heading_docs[h].add(doc_id)
            if h not in global_heading_level or level < global_heading_level[h]:
                global_heading_level[h] = level

        # Vector accumulation
        if emb_blob is not None:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if vec.size > 0:
                if doc_id not in doc_vector_sums:
                    doc_vector_sums[doc_id] = vec.astype(np.float64, copy=True)
                else:
                    doc_vector_sums[doc_id] += vec
                doc_vector_counts[doc_id] += 1

    # Mean-pool and L2-normalize document vectors
    embedded_doc_ids: List[int] = []
    embedded_vectors: List[np.ndarray] = []
    for doc_id, _path, _hash in docs_raw:
        v_count = doc_vector_counts.get(doc_id, 0)
        if v_count > 0 and doc_id in doc_vector_sums:
            mean_vec = doc_vector_sums[doc_id] / float(v_count)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-12:
                embedded_doc_ids.append(doc_id)
                embedded_vectors.append((mean_vec / norm).astype(np.float32))

    if embedded_vectors:
        X = np.vstack(embedded_vectors)
        raw_labels, chosen_k = cluster_documents(X, seed=seed)
    else:
        raw_labels = np.empty(0, dtype=np.int32)
        chosen_k = 0

    # Group embedded documents by raw cluster index
    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for doc_id, raw_c in zip(embedded_doc_ids, raw_labels.tolist()):
        cluster_members[int(raw_c)].append(doc_id)

    # Sort clusters by document_count descending (tie-break by chunk_count desc, min doc_id asc)
    sorted_raw_clusters = sorted(
        cluster_members.keys(),
        key=lambda c: (
            -len(cluster_members[c]),
            -sum(doc_chunk_counts.get(d, 0) for d in cluster_members[c]),
            min(cluster_members[c]),
        ),
    )

    doc_path_map = {doc_id: path for doc_id, path, _ in docs_raw}
    doc_topic_map: Dict[int, Optional[str]] = {
        doc_id: None for doc_id, _, _ in docs_raw
    }
    topics_list: List[Dict[str, Any]] = []

    for topic_idx, raw_c in enumerate(sorted_raw_clusters):
        topic_id = f"t{topic_idx}"
        member_ids = cluster_members[raw_c]
        for d_id in member_ids:
            doc_topic_map[d_id] = topic_id

        cluster_chunk_total = sum(doc_chunk_counts.get(d_id, 0) for d_id in member_ids)

        # Score distinct H1s by member document coverage, then cluster chunk count, then shortest text
        h1_doc_cov: Counter[str] = Counter()
        h1_chunk_sum: Counter[str] = Counter()
        h12_doc_cov: Counter[str] = Counter()
        h12_chunk_sum: Counter[str] = Counter()

        for d_id in member_ids:
            for h, c_cnt in doc_h1_chunk_counts.get(d_id, {}).items():
                h1_doc_cov[h] += 1
                h1_chunk_sum[h] += c_cnt
            for h, c_cnt in doc_h12_chunk_counts.get(d_id, {}).items():
                h12_doc_cov[h] += 1
                h12_chunk_sum[h] += c_cnt

        if h1_doc_cov:
            ranked_h1 = sorted(
                h1_doc_cov.keys(),
                key=lambda h: (-h1_doc_cov[h], -h1_chunk_sum[h], len(h), h),
            )
            winner_label = ranked_h1[0]
        elif h12_doc_cov:
            ranked_h12_fallback = sorted(
                h12_doc_cov.keys(),
                key=lambda h: (-h12_doc_cov[h], -h12_chunk_sum[h], len(h), h),
            )
            winner_label = ranked_h12_fallback[0]
        else:
            # Fallback to most common filename stem in the cluster
            stem_counts = Counter(Path(doc_path_map[d_id]).stem for d_id in member_ids)
            winner_label = sorted(
                stem_counts.keys(),
                key=lambda s: (-stem_counts[s], len(s), s),
            )[0]

        ranked_h12 = sorted(
            h12_doc_cov.keys(),
            key=lambda h: (-h12_doc_cov[h], -h12_chunk_sum[h], len(h), h),
        )
        label_headings = [winner_label]
        for h in ranked_h12:
            if h not in label_headings:
                label_headings.append(h)
            if len(label_headings) >= 3:
                break

        topics_list.append(
            {
                "id": topic_id,
                "label": winner_label,
                "label_headings": label_headings,
                "document_count": len(member_ids),
                "chunk_count": cluster_chunk_total,
            }
        )

    # Assemble roots list
    roots_payload: List[Dict[str, Any]] = []
    for rdef in root_defs:
        rid = rdef["id"]
        roots_payload.append(
            {
                "id": rid,
                "label": rdef["label"],
                "path_prefix": rdef["path_prefix"],
                "document_count": root_doc_counts.get(rid, 0),
                "chunk_count": root_chunk_counts.get(rid, 0),
            }
        )

    if root_doc_counts.get("other", 0) > 0 or not roots_payload:
        if root_doc_counts.get("other", 0) > 0:
            roots_payload.append(
                {
                    "id": "other",
                    "label": "other",
                    "path_prefix": "",
                    "document_count": root_doc_counts["other"],
                    "chunk_count": root_chunk_counts["other"],
                }
            )

    # Assemble top 80 global H1/H2 headings
    sorted_global_headings = sorted(
        global_heading_chunk_counts.keys(),
        key=lambda h: (
            -global_heading_chunk_counts[h],
            -len(global_heading_docs[h]),
            global_heading_level.get(h, 2),
            h,
        ),
    )[:MAX_GLOBAL_HEADINGS]

    headings_payload = [
        {
            "text": h,
            "level": global_heading_level.get(h, 1),
            "chunk_count": int(global_heading_chunk_counts[h]),
            "document_ids": sorted(global_heading_docs[h]),
        }
        for h in sorted_global_headings
    ]

    # Assemble documents list
    documents_payload: List[Dict[str, Any]] = []
    for doc_id, path, _content_hash in docs_raw:
        title = doc_first_h1.get(doc_id) or Path(path).stem
        h_counts = doc_heading_counts.get(doc_id, Counter())
        first_seen = doc_heading_first_seen.get(doc_id, {})
        top_headings = sorted(
            h_counts.keys(),
            key=lambda h: (-h_counts[h], first_seen.get(h, 0)),
        )[:MAX_DOC_TOP_HEADINGS]

        documents_payload.append(
            {
                "id": doc_id,
                "path": path,
                "title": title,
                "root_id": doc_root_map[doc_id],
                "topic_id": doc_topic_map.get(doc_id),
                "top_headings": top_headings,
            }
        )

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "built_at": built_at,
        "model_name": model_name,
        "stats": {
            "documents": len(docs_raw),
            "chunks": len(chunks_raw),
        },
        "clustering": {
            "k": chosen_k,
            "seed": seed,
        },
        "roots": roots_payload,
        "headings": headings_payload,
        "topics": topics_list,
        "documents": documents_payload,
    }
    return payload, source_fingerprint


def ensure_overview(
    storage: Storage,
    include_paths: Sequence[str],
    model_name: str,
    force: bool = False,
    seed: int = DEFAULT_CLUSTER_SEED,
) -> Dict[str, Any]:
    """
    Returns the cached overview payload if it matches schema_version, model_name, and
    source_fingerprint; otherwise builds, persists, and returns a fresh overview payload.
    """
    fp_rows = storage.get_corpus_fingerprint_rows()
    current_fp = compute_source_fingerprint(fp_rows)

    if not force:
        existing = storage.get_overview()
        if (
            existing is not None
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("model_name") == model_name
            and existing.get("source_fingerprint") == current_fp
            and isinstance(existing.get("payload"), dict)
        ):
            return existing["payload"]

    payload, computed_fp = build_overview_payload(
        storage=storage,
        include_paths=include_paths,
        model_name=model_name,
        seed=seed,
    )
    storage.save_overview(
        schema_version=SCHEMA_VERSION,
        model_name=model_name,
        built_at=payload["built_at"],
        source_fingerprint=computed_fp,
        payload=payload,
    )
    return payload


PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#d2a8ff",
    "#f0883e",
    "#79c0ff",
    "#56d364",
    "#ff7b72",
    "#e3b341",
    "#a5d6ff",
    "#bc8cff",
    "#ffa657",
    "#39d353",
]


def compute_treemap_rects(
    items: Sequence[Dict[str, Any]],
    weight_key: str,
    width: float = 800.0,
    height: float = 180.0,
    gap: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Computes 2D treemap bounding boxes (x, y, w, h) using recursive balanced binary partitioning.
    Preserves item metadata and attaches x, y, w, h, color, and share_percent.
    """
    positive = [dict(item) for item in items if item.get(weight_key, 0) > 0]
    if not positive:
        positive = [dict(item) for item in items]
    if not positive:
        return []

    actual_total = sum(int(it.get(weight_key, 0)) for it in positive)

    for idx, it in enumerate(positive):
        it["_weight"] = float(max(1, int(it.get(weight_key, 0))))
        raw_val = int(it.get(weight_key, 0))
        it["share_percent"] = (
            round((raw_val / actual_total) * 100.0, 1) if actual_total > 0 else 0.0
        )
        it["color"] = PALETTE[idx % len(PALETTE)]

    def _partition(
        sub_items: List[Dict[str, Any]], x: float, y: float, w: float, h: float
    ) -> None:
        if not sub_items:
            return
        if len(sub_items) == 1:
            it = sub_items[0]
            it["x"] = round(x, 1)
            it["y"] = round(y, 1)
            it["w"] = round(max(12.0, w), 1)
            it["h"] = round(max(12.0, h), 1)
            return

        total = sum(i["_weight"] for i in sub_items)
        half = total / 2.0
        running = 0.0
        split_idx = 1
        for i in range(len(sub_items) - 1):
            running += sub_items[i]["_weight"]
            split_idx = i + 1
            if running >= half:
                break

        left = sub_items[:split_idx]
        right = sub_items[split_idx:]
        left_ratio = sum(i["_weight"] for i in left) / total if total > 0 else 0.5

        if w >= h:
            avail_w = max(24.0, w - gap)
            w_left = avail_w * left_ratio
            w_right = avail_w - w_left
            _partition(left, x, y, w_left, h)
            _partition(right, x + w_left + gap, y, w_right, h)
        else:
            avail_h = max(24.0, h - gap)
            h_left = avail_h * left_ratio
            h_right = avail_h - h_left
            _partition(left, x, y, w, h_left)
            _partition(right, x, y + h_left + gap, w, h_right)

    sorted_items = sorted(positive, key=lambda i: -i["_weight"])
    for idx, it in enumerate(sorted_items):
        it["color"] = PALETTE[idx % len(PALETTE)]

    _partition(sorted_items, 0.0, 0.0, width, height)
    for it in sorted_items:
        it.pop("_weight", None)
    return sorted_items


def build_overview_view_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches the stored overview payload with SVG layout coordinates and font sizes for templates."""
    roots = payload.get("roots", [])
    topics = payload.get("topics", [])
    headings = payload.get("headings", [])
    documents = payload.get("documents", [])

    root_rects = compute_treemap_rects(
        roots, weight_key="document_count", width=800.0, height=160.0, gap=6.0
    )
    topic_rects = compute_treemap_rects(
        topics, weight_key="document_count", width=800.0, height=180.0, gap=5.0
    )

    topic_color_map = {t["id"]: t["color"] for t in topic_rects}
    max_topic_docs = (
        max((int(t.get("document_count", 0)) for t in topics), default=1) or 1
    )
    total_docs = max(1, int(payload.get("stats", {}).get("documents", 0)))

    enriched_topics = []
    for idx, t in enumerate(topics):
        t_copy = dict(t)
        doc_cnt = int(t.get("document_count", 0))
        t_copy["color"] = topic_color_map.get(t["id"], PALETTE[idx % len(PALETTE)])
        t_copy["bar_percent"] = round((doc_cnt / max_topic_docs) * 100.0, 1)
        t_copy["share_percent"] = round((doc_cnt / total_docs) * 100.0, 1)
        enriched_topics.append(t_copy)

    # Scale heading font sizes between 0.82rem and 1.50rem
    if headings:
        counts = [int(h.get("chunk_count", 1)) for h in headings]
        min_c = min(counts)
        max_c = max(counts)
    else:
        min_c, max_c = 1, 1

    enriched_headings = []
    for h in headings:
        h_copy = dict(h)
        c = int(h.get("chunk_count", 1))
        if max_c > min_c:
            norm = (np.log1p(c) - np.log1p(min_c)) / (np.log1p(max_c) - np.log1p(min_c))
        else:
            norm = 0.5
        font_rem = round(0.82 + float(norm) * 0.68, 2)
        h_copy["font_size_rem"] = font_rem
        h_copy["doc_count"] = len(h.get("document_ids", []))
        enriched_headings.append(h_copy)

    root_label_map = {r["id"]: r["label"] for r in roots}
    topic_label_map = {t["id"]: t["label"] for t in topics}

    enriched_docs = []
    for d in documents:
        d_copy = dict(d)
        tid = d.get("topic_id")
        rid = d.get("root_id")
        d_copy["root_label"] = root_label_map.get(rid, rid or "other")
        d_copy["topic_label"] = (
            topic_label_map.get(tid, "Unclustered") if tid else "Unclustered"
        )
        d_copy["topic_color"] = (
            topic_color_map.get(tid, "#8b949e") if tid else "#8b949e"
        )
        enriched_docs.append(d_copy)

    return {
        "overview": payload,
        "root_rects": root_rects,
        "topic_rects": topic_rects,
        "topics": enriched_topics,
        "headings": enriched_headings,
        "documents": enriched_docs,
    }
