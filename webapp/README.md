---
title: AMI Risk Predictor
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict AMI from blood-panel labs using a top-K model bundle trained by final.py.
---

# AMI Risk Predictor

A clinical-style web demo that predicts the probability of **Acute Myocardial
Infarction (AMI)** from a routine blood panel. Built on top of the
`final.py` training suite (18 baseline + cutting-edge models); the top
**K** models by AUC-ROC are bundled and served live.

> **Research use only.** This is not a medical device and must not be used
> for clinical diagnosis or treatment.

## Architecture

```
final.py            ─►  final_assets/bundle.joblib   (scaler + top-K fitted models)
                        final_assets/metrics.json    (leaderboard, feature ranges)

webapp/
  backend/main.py        FastAPI — /api/features /api/metrics /api/predict
  frontend/              Next.js 14 (static export)
  Dockerfile             Multi-stage build, listens on :7860 (HF Spaces port)
```

The Docker image builds the Next.js static export, copies it into the
FastAPI image, and serves both the API and the UI from a single origin.

## Local development

```bash
# 1. Train the models and write the serving bundle
python final.py

# 2. Run the FastAPI backend
cd webapp/backend
uvicorn main:app --reload   # http://localhost:8000

# 3. Run the Next.js frontend in dev mode (separate terminal)
cd webapp/frontend
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

## Production build (used by HF Spaces)

```bash
docker build -t ami-predictor webapp/
docker run -p 7860:7860 ami-predictor
# Open http://localhost:7860
```

## Configuration

| Constant | Where | Default | Effect |
| --- | --- | --- | --- |
| `TOP_K_TO_SERVE` | `final.py` | `5` | How many models from the leaderboard get bundled and served. Set to `18` to expose every trained model. |
| `SERVING_BLOCKLIST` | `final.py` | `{"AutoGluon"}` | Models excluded from the served bundle even if they rank high (AutoGluon's predictor is not joblib-friendly and inflates the image). |
| `AMI_ASSETS_DIR` | env var | unset | Override where the backend looks for `bundle.joblib` / `metrics.json`. |
