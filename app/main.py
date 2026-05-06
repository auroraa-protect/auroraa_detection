from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import detection_router

app = FastAPI(
    title="Auroraa Forensics API",
    description="API for image manipulation detection",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the forensics router
app.include_router(detection_router)

@app.get("/")
def root():
    return {"message": "Auroraa Forensics API is running. Send POST requests to /api/analyze"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
