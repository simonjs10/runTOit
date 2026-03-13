"""
main.py – production entry point for Render / Railway / Fly.io

Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

# Re-export the FastAPI app defined in the notebook's extracted source.
# When deploying without Jupyter, copy the notebook cells into this file
# or import the app object from your extracted module.
from route_agent import app
import uvicorn

# ── If you have extracted the notebook to route_agent.py ──────────────────
# from route_agent import app

# ── Otherwise paste the notebook source here and run directly ─────────────
if __name__ == "__main__":
    # Import app from wherever you have placed the notebook source
    from route_agent import app  # noqa: F401  adjust import as needed
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
