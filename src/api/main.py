from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import attendance, admin, enrollment
from src.api.auth import router as auth_router
from src.infrastructure.database import init_db

# Initialize DB tables (SQLite creates the file automatically)
init_db()

app = FastAPI(
    title="OneApp Voice Biometric API",
    description="Phase 10 POC API with Security, JWT Auth, Database, and Enrollment.",
    version="1.1.0"
)

# 8. CORS MUST NOT BE PERMISSIVE BY DEFAULT
origins = [
    # Add Android emulator or specific test IPs here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(attendance.router, prefix="/api/v1")
app.include_router(enrollment.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "phase": 10}

