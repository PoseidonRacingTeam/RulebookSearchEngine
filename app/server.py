"""FastAPI server for Rulebook Search Engine (localhost only by default)."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .search import SearchEngine

log = logging.getLogger("fs.server")
APP_SIGNATURE = "formula-student-rule-search"

engine = SearchEngine(config.INDEX_DIR, config.PDF_PATH, config.MODELS_DIR)
_pending = 0
_pending_lock = threading.Lock()
_started_at = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=engine.load, name="engine-load", daemon=True)
    t.start()
    yield


app = FastAPI(title="Rulebook Search Engine", lifespan=lifespan, docs_url=None, redoc_url=None)

if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
if (config.INDEX_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(config.INDEX_DIR / "assets")), name="assets")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/pdf")
def pdf():
    if not config.PDF_PATH.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(config.PDF_PATH, media_type="application/pdf", headers={"Content-Disposition": "inline; filename=FS_Rules.pdf"})


@app.get("/api/status")
def status():
    s = engine.status()
    s["app"] = APP_SIGNATURE
    s["uptime_s"] = round(time.time() - _started_at, 1)
    return s


@app.get("/api/search")
def search(q: str = Query(""), scope: str = Query("all")):
    global _pending
    if not engine.ready:
        payload = {"type": "error", "message": "The search engine is not ready yet." if engine.state != "error" else f"The search engine failed to start: {engine.error}", "status": engine.status()}
        return JSONResponse(payload, status_code=503)
    with _pending_lock:
        if _pending >= config.MAX_PENDING_SEARCHES:
            return JSONResponse({"type": "error", "message": "The server is busy with earlier searches. Please wait a moment and try again."}, status_code=429)
        _pending += 1
    try:
        res = engine.search(q, scope)
    except Exception as e:  # noqa: BLE001
        log.exception("search failed")
        res = {"type": "error", "message": f"Search failed: {type(e).__name__}: {e}"}
    finally:
        with _pending_lock:
            _pending -= 1
    return res


@app.get("/api/rule/{code}")
def rule(code: str):
    if not engine.ready:
        raise HTTPException(503, "not ready")
    r = engine.get_rule(code)
    if not r:
        raise HTTPException(404, f"No rule {code}")
    return r


@app.get("/api/section/{code}")
def section(code: str):
    if not engine.ready:
        raise HTTPException(503, "not ready")
    rules = engine.section_rules(code)
    if not rules:
        raise HTTPException(404, f"No section {code}")
    head = engine.rules_by_key.get(code.replace(" ", "").upper())
    return {"code": head["code"] if head else code, "title": head.get("title", "") if head else "", "kind": head["kind"] if head else "", "rules": rules}


@app.get("/api/glossary")
def glossary():
    if not engine.ready:
        raise HTTPException(503, "not ready")
    return engine.data["glossary"]


@app.middleware("http")
async def no_cache_api(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"   # always revalidate so UI changes show up on a normal reload
    return resp
