from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fuzzy_nlp import find_matches  # Make sure this is implemented correctly

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (helpful for Power BI or web apps calling this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class SearchRequest(BaseModel):
    query: str

# Define POST endpoint
@app.post("/search")
def search(request: SearchRequest):
    return find_matches(request.query)

# Local run entry point (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
