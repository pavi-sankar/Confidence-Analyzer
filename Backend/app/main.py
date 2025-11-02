from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.v1.v1_router import frontend_router

# create app
app = FastAPI()

# CORS configuration — change origins to your frontend in production
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Correct: add CORSMiddleware with add_middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok"}

app.include_router(frontend_router)
