from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.agent_routes import router as agent_router
from backend.app.api.game_routes import router as game_router

app = FastAPI(title="Connect Four RL Lab", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(game_router)
app.include_router(agent_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
