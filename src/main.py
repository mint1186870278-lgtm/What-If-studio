"""FastAPI application entry point"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.db import init_db
from src.core.anet_gateway import register_anet_services, unregister_anet_services

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 Starting whatif-studio API")
    settings.validate_openai_env()
    settings.ensure_storage_paths()
    # Ensure Chroma persistence directory
    from pathlib import Path as _Path
    _Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info("✅ Database initialized")

    # Register services on the ANet P2P mesh (if daemon is running)
    registered = await register_anet_services()
    if registered:
        logger.info("✅ Registered %d ANet services", len(registered))
    else:
        logger.info(
            "ANet daemon not detected — ANet service exposure is disabled in this run. "
            "Install anet CLI (curl -fsSL https://agentnetwork.org.cn/install.sh | sh) "
            "and run 'anet daemon &' to enable P2P agent mesh."
        )

    yield

    # Shutdown
    logger.info("🛑 Shutting down whatif-studio API")
    await unregister_anet_services()


# Create FastAPI application
app = FastAPI(
    title="whatif-studio API",
    description="FastAPI backend for whatif-studio creative video editing",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "whatif-studio"}


@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "ok", "service": "whatif-studio"}


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include API routes
from src.api import projects, assets, sessions, jobs, gateway, agents, ws, feedback

app.include_router(projects.router, prefix="/api", tags=["projects"])
app.include_router(assets.router, prefix="/api", tags=["assets"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(gateway.router, prefix="/api", tags=["gateway"])  # Gateway routes
app.include_router(agents.router, prefix="/api", tags=["agents"])
app.include_router(ws.router, tags=["websocket"])  # WebSocket for human-in-the-loop
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


def resolve_frontend_dist_dir() -> Path:
    return settings.frontend_dist_dir.resolve()


def mount_frontend_assets(application: FastAPI) -> None:
    dist_dir = resolve_frontend_dist_dir()
    if not dist_dir.exists():
        logger.warning("Frontend dist directory not found: %s", dist_dir)
        return
    for folder in ("assets", "mock", "background"):
        target = dist_dir / folder
        if target.exists():
            application.mount(f"/{folder}", StaticFiles(directory=target), name=f"frontend-{folder}")


mount_frontend_assets(app)


@app.get("/", include_in_schema=False)
async def serve_frontend_index():
    dist_dir = resolve_frontend_dist_dir()
    index_file = dist_dir / "index.html"
    if not index_file.exists():
        raise RuntimeError(f"Frontend index not found: {index_file}")
    return FileResponse(index_file)


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    if full_path.startswith("api/") or full_path == "health" or full_path == "api/health":
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    dist_dir = resolve_frontend_dist_dir()
    file_candidate = (dist_dir / full_path).resolve()
    if dist_dir in file_candidate.parents and file_candidate.exists() and file_candidate.is_file():
        return FileResponse(file_candidate)

    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"detail": "Frontend not built"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
