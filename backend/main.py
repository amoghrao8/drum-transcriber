from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import audio, analysis, teacher

app = FastAPI(title="DrumScribe AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audio.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(teacher.router, prefix="/api")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "DrumScribe AI"}
