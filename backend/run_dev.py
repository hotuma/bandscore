import uvicorn
import os

if __name__ == "__main__":
    # Ensure workers=1 to support in-memory job store
    print("Starting BandScore Backend (Single Worker Mode)...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1)
