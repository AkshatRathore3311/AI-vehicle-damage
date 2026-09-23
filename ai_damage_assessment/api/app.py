from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .routes import router
from .schemas import HealthResponseSchema

app = FastAPI(
    title="AutoDamage AI - Vehicle Damage Assessment & Cost Estimator",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

WEB_HTML_PATH = Path(__file__).parent.parent / "web" / "index.html"

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_web_ui():
    if WEB_HTML_PATH.exists():
        with open(WEB_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AutoDamage AI Web Dashboard</h1>"

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health", response_model=HealthResponseSchema, tags=["System Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Vehicle Damage Assessment Service",
        "version": "1.0.0",
        "model_status": "loaded",
        "device": "cpu"
    }
