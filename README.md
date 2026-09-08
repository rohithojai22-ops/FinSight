# FinSight - Financial Behavior Intelligence Platform

## Tech Stack
- Frontend: HTML, CSS, JavaScript, Chart.js
- Backend: FastAPI + Python
- Database: PostgreSQL
- Analytics: Pandas, NumPy
- ML: Isolation Forest
- Forecasting: Trend Regression + Moving Average

## Database Setup

This project now uses PostgreSQL instead of SQLite.

Set the environment variable:

DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE?sslmode=require

For local development, copy `backend/.env.example` values into your system environment or configure DATABASE_URL in your deployment platform.

## Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload
```

## Deployment

Recommended:
- Frontend: GitHub Pages or Netlify
- Backend: Render
- Database: Supabase PostgreSQL or Neon PostgreSQL

On Render, add DATABASE_URL as an environment variable.
