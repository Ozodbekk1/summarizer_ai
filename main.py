import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"Starting Summarizer AI API on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False)
