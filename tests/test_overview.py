import numpy as np
import pytest
from fastapi.testclient import TestClient

from rag_search_engine.core.overview import (
    SCHEMA_VERSION,
    build_overview_payload,
    build_overview_view_context,
    ensure_overview,
    parse_chunk_breadcrumb,
)
from rag_search_engine.core.store import Storage
import rag_search_engine.web.main as web_main


def _unit_vec(dim: int, hot_idx: int, noise: float = 0.02, rng=None) -> list[float]:
    if rng is None:
        rng = np.random.default_rng(42)
    v = rng.normal(0.0, noise, size=dim).astype(np.float32)
    v[hot_idx % dim] += 1.0
    norm = float(np.linalg.norm(v))
    return (v / norm).tolist()


@pytest.fixture
def populated_store(tmp_path):
    """Creates a SQLite store with 30 documents across 3 roots + 1 document without vectors."""
    db_path = tmp_path / "overview_test.db"
    store = Storage(str(db_path))
    rng = np.random.default_rng(123)

    roots = [
        "/kb/_context",
        "/kb/_condensed",
        "/kb/LLM-RAG",
    ]

    theme_headings = [
        ("Hybrid Search", "Reciprocal Rank Fusion", "BM25 Details"),
        ("Vector Databases", "SQLite Vec", "Cosine Distance"),
        ("Chunking Strategy", "Markdown AST", "Token Limits"),
        ("Agent Workflows", "Tool Calling", "Error Recovery"),
        ("Evaluation Metrics", "Recall At K", "NDCG"),
        ("Prompt Engineering", "System Instructions", "Few Shot"),
        ("Embeddings", "MiniLM Architecture", "Pooling"),
        ("Graph RAG", "Knowledge Graphs", "Entity Extraction"),
        ("Local Inference", "CPU Optimization", "Quantization"),
        ("Web UI", "HTMX Patterns", "FastAPI Templates"),
    ]

    # Insert 30 documents with vectors across 10 tight themes
    for i in range(30):
        theme_idx = i % len(theme_headings)
        h1, h2, h3 = theme_headings[theme_idx]
        root = roots[0] if i < 20 else (roots[1] if i < 26 else roots[2])
        doc_path = f"{root}/doc_{i:02d}.md"

        vec1 = _unit_vec(384, theme_idx * 10, noise=0.01, rng=rng)
        vec2 = _unit_vec(384, theme_idx * 10, noise=0.01, rng=rng)
        chunks = [
            (f"[{h1} > {h2}] | Content paragraph 1 for doc {i}.", vec1),
            (f"[{h1} > {h2} > {h3}] | Content paragraph 2 for doc {i}.", vec2),
        ]
        store.add_document(doc_path, f"hash_{i}", chunks)

    # Also insert 1 document outside configured roots and with no chunks/vectors
    with store.conn:
        store.conn.execute(
            "INSERT INTO documents (path, content_hash) VALUES (?, ?)",
            ("/unmatched/empty_doc.md", "hash_empty"),
        )

    return store, roots


def test_parse_chunk_breadcrumb():
    assert parse_chunk_breadcrumb("[Intro > Setup > Step 1] | Run make") == (
        "Intro",
        "Setup",
        "Step 1",
    )
    assert parse_chunk_breadcrumb("[Intro > ... > Deep H3] | Skipped H2") == (
        "Intro",
        None,
        "Deep H3",
    )
    assert parse_chunk_breadcrumb("Plain text without breadcrumb") == (
        None,
        None,
        None,
    )


def test_prd_acceptance_criteria(populated_store):
    store, include_paths = populated_store
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    payload = ensure_overview(store, include_paths, model_name)

    # 1. Validates against schema version 1
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["model_name"] == model_name
    assert "built_at" in payload
    assert 8 <= payload["clustering"]["k"] <= 24
    assert payload["clustering"]["seed"] == 7

    # 2. stats.documents equals COUNT(*) from documents, and every documents.id exists in SQLite
    db_doc_ids = {
        row[0] for row in store.conn.execute("SELECT id FROM documents").fetchall()
    }
    assert payload["stats"]["documents"] == len(db_doc_ids)
    payload_doc_ids = {d["id"] for d in payload["documents"]}
    assert payload_doc_ids == db_doc_ids

    # 3. Every document with >=1 vector has non-null topic_id; union of topic members equals that set
    docs_with_vecs = {
        row[0]
        for row in store.conn.execute(
            "SELECT DISTINCT c.doc_id FROM chunks c JOIN vec_chunks v ON v.rowid = c.id"
        ).fetchall()
    }
    assigned_docs = {d["id"] for d in payload["documents"] if d["topic_id"] is not None}
    unassigned_docs = {d["id"] for d in payload["documents"] if d["topic_id"] is None}
    assert assigned_docs == docs_with_vecs
    assert unassigned_docs == (db_doc_ids - docs_with_vecs)
    assert sum(t["document_count"] for t in payload["topics"]) == len(docs_with_vecs)

    # 4. Root document counts sum to stats.documents
    assert (
        sum(r["document_count"] for r in payload["roots"])
        == payload["stats"]["documents"]
    )
    root_ids = {r["id"] for r in payload["roots"]}
    assert {"context", "condensed", "llm-rag", "other"} <= root_ids

    # 5. Rebuilding twice on an unchanged index yields the same topics[].id assignment and labels
    payload_rebuilt, _ = build_overview_payload(store, include_paths, model_name)
    assert [(t["id"], t["label"], t["document_count"]) for t in payload["topics"]] == [
        (t["id"], t["label"], t["document_count"]) for t in payload_rebuilt["topics"]
    ]
    assert [(d["id"], d["topic_id"], d["title"]) for d in payload["documents"]] == [
        (d["id"], d["topic_id"], d["title"]) for d in payload_rebuilt["documents"]
    ]

    # 6. Changing one file's content_hash changes source_fingerprint and causes rebuild
    stored_before = store.get_overview()
    assert stored_before is not None
    fp_before = stored_before["source_fingerprint"]

    # Calling ensure_overview when unchanged keeps the exact stored row
    cached_payload = ensure_overview(store, include_paths, model_name)
    assert cached_payload["built_at"] == stored_before["built_at"]

    # Modify one document's content_hash
    with store.conn:
        store.conn.execute(
            "UPDATE documents SET content_hash = 'modified_hash' WHERE path = '/kb/_context/doc_00.md'"
        )
    ensure_overview(store, include_paths, model_name)
    stored_after = store.get_overview()
    assert stored_after is not None
    assert stored_after["source_fingerprint"] != fp_before

    # Changing model_name also triggers a rebuild
    ensure_overview(store, include_paths, "custom/new-model")
    stored_model_change = store.get_overview()
    assert stored_model_change is not None
    assert stored_model_change["model_name"] == "custom/new-model"

    # 7. Heading text shown in the payload occurs verbatim in some chunk breadcrumb
    all_breadcrumbs = set()
    for (content,) in store.conn.execute("SELECT content FROM chunks").fetchall():
        for seg in parse_chunk_breadcrumb(content):
            if seg:
                all_breadcrumbs.add(seg)

    for h in payload["headings"]:
        assert h["text"] in all_breadcrumbs
        assert h["level"] in (1, 2)


def test_overview_web_endpoints(populated_store, monkeypatch):
    store, include_paths = populated_store
    monkeypatch.setattr(web_main, "_get_storage", lambda: store)
    monkeypatch.setattr(web_main.settings.indexing, "include_paths", include_paths)
    monkeypatch.setattr(
        web_main.settings.search,
        "model_name",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    client = TestClient(web_main.app)

    # Index page includes the Overview tab
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "Overview &amp; Topic Clusters" in res_index.text

    # JSON API endpoint
    res_api = client.get("/api/overview")
    assert res_api.status_code == 200
    data = res_api.json()
    assert data["schema_version"] == 1
    assert data["stats"]["documents"] == 31

    # HTML panel endpoint
    res_html = client.get("/overview")
    assert res_html.status_code == 200
    assert "Roots Distribution" in res_html.text
    assert "Topic Clusters" in res_html.text
    assert "Heading Cloud" in res_html.text
    assert "Hybrid Search" in res_html.text

    # Force rebuild endpoint
    res_rebuild = client.post("/overview/rebuild")
    assert res_rebuild.status_code == 200
    assert "Topic Clusters" in res_rebuild.text

    # View context helper check
    ctx = build_overview_view_context(data)
    assert len(ctx["root_rects"]) == 4
    assert len(ctx["topic_rects"]) == len(data["topics"])
