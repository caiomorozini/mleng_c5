from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
from fastapi import FastAPI
from app.config import model_config


app = FastAPI(
    title="Datathon",
    description="Case passos mágicos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Carrega modelo e artefatos na inicialização"""
    model_config.load_artifacts()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
