import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.search_controller import SearchController, normalize_scores
from rag_search_engine.web.core.config import get_settings

query = "scatter plot"

settings = get_settings()
embedder = Embedder(model_name=settings.search.model_name)
storage = Storage(db_path=settings.storage.db_path)
controller = SearchController(embedder, storage, threshold=settings.search.threshold)

# --- 1. FTS results ---
fts_q = storage._sanitize_fts_query(query)
print(f"FTS query string: {fts_q!r}\n")

fts_results = storage.search_fts(query, limit=5)
print(f"FTS results ({len(fts_results)}):")
for path, content, rank in fts_results:
    print(f"  rank={rank:.4f}")
    print(f"  path={path}")
    print(f"  content={content[:100]!r}\n")

# --- 2. Vector results (unfiltered, show all top-5) ---
qvec = embedder.embed(query)[0].tolist()
vec_raw = storage.search_vector(qvec, limit=5)
print(f"Vector results ({len(vec_raw)}) [threshold={settings.search.threshold}]:")
for path, content, distance in vec_raw:
    score = 1.0 - (distance**2 / 2.0)
    flag = "PASS" if score >= settings.search.threshold else "below threshold"
    print(f"  distance={distance:.4f}  score={score:.4f}  [{flag}]")
    print(f"  path={path}")
    print(f"  content={content[:100]!r}\n")

# --- 3. Final fused results ---
results = controller.search(query, limit=5)
norm = normalize_scores(results)
print(f"Final results ({len(norm)}):")
for path, content, score in norm:
    print(f"  relevance={score:.0f}%")
    print(f"  path={path}")
    print(f"  content={content[:100]!r}\n")
