from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.analyzer.router import router as analyzer_router

app = FastAPI(title="Ski Motion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyzer_router)

@app.get("/")
async def root():
    return {"status": "running"}
