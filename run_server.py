"""Server launcher for Uvicorn ASGI Server."""
import uvicorn

if __name__ == "__main__":
    print("Starting AI Vehicle Damage Assessment API on http://127.0.0.1:8000 ...")
    uvicorn.run("ai_damage_assessment.api.app:app", host="127.0.0.1", port=8000, reload=False)
