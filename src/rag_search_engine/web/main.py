from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.chunker import MarkdownChunker
from rag_search_engine.core.search_controller import SearchController, normalize_scores
from rag_search_engine.web.core.config import get_settings

# Global singletons
controller: Optional[SearchController] = None
chunker: Optional[MarkdownChunker] = None
settings = get_settings()

# Progress state
sync_progress = {"current": 0, "total": 0, "status": "idle"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global controller, chunker
    embedder = Embedder(model_name=settings.search.model_name)
    storage = Storage(db_path=settings.storage.db_path)
    chunker = MarkdownChunker(model_name=settings.search.model_name)
    controller = SearchController(
        embedder=embedder, storage=storage, threshold=settings.search.threshold
    )
    yield


app = FastAPI(lifespan=lifespan)

# Setup templates
base_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_path / "templates"))


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from fastapi.responses import HTMLResponse, PlainTextResponse
import os

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(None)):
    if not query:
        return templates.TemplateResponse(
            "search_results.html", {"request": request, "results": []}
        )

    results = controller.search(query, limit=10)
    return templates.TemplateResponse(
        "search_results.html",
        {"request": request, "results": normalize_scores(results)},
    )


@app.get("/api/document", response_class=PlainTextResponse)
async def get_document(path: str):
    try:
        if not os.path.exists(path):
            return PlainTextResponse("File not found.", status_code=404)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return PlainTextResponse(f"Error reading document: {e}", status_code=500)


def background_sync():
    global sync_progress
    sync_progress["status"] = "indexing"
    sync_progress["current"] = 0

    # 1. Pre-calculate total files
    all_files = []
    for path in settings.indexing.include_paths:
        p = Path(path)
        if p.is_dir():
            all_files.extend(list(p.rglob("*.md")))
        elif p.is_file() and p.suffix == ".md":
            all_files.append(p)

    total = len(all_files)
    sync_progress["total"] = total

    if total == 0:
        sync_progress["status"] = "idle"
        return

    # 2. Index files one by one to keep progress accurate across multiple paths
    for i, file_path in enumerate(all_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"DEBUG: Indexing {file_path} (len={len(content)})")
            controller.index_file(str(file_path), content, chunker)
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")

        sync_progress["current"] = i + 1

    sync_progress["status"] = "idle"


@app.post("/sync", response_class=HTMLResponse)
async def start_sync(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(background_sync)
    return templates.TemplateResponse(
        "sync_status.html",
        {"request": request, "current": 0, "total": 0, "progress_percent": 0},
    )


@app.get("/sync/status", response_class=HTMLResponse)
async def get_sync_status(request: Request):
    current = sync_progress["current"]
    total = sync_progress["total"]
    percent = int((current / total * 100) if total > 0 else 0)

    return templates.TemplateResponse(
        "sync_status.html",
        {
            "request": request,
            "current": current,
            "total": total,
            "progress_percent": percent,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
