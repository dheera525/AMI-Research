# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# AMI Risk Predictor — single-image build for Hugging Face Spaces
#   stage 1: build the Next.js static export
#   stage 2: Python + FastAPI runtime that serves /api/* and the static site
# ---------------------------------------------------------------------------

# 1) Frontend build
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
# Next 14 with `output: "export"` writes the static site to ./out
RUN npm run build

# 2) Python runtime
FROM python:3.11-slim
WORKDIR /app

# libgomp1 is required by lightgbm/xgboost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend code
COPY backend/ ./backend/
# Static frontend (mounted by FastAPI as "/")
COPY --from=web /web/out ./backend/static/
# Trained models from `python final.py`
COPY final_assets/ ./final_assets/

ENV AMI_ASSETS_DIR=/app/final_assets
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
