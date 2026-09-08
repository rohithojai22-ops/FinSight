import os
import io
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add your PostgreSQL connection string as an environment variable."
    )

app = FastAPI(title="FinSight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Tx(BaseModel):
    date: str
    description: str
    category: str
    amount: float
    type: str


class Goal(BaseModel):
    name: str
    target_amount: float
    current_savings: float
    monthly_income: float
    months_remaining: int


@contextmanager
def con():
    connection = psycopg2.connect(DATABASE_URL)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@app.on_event("startup")
def init():
    with con() as c:
        cur = c.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                type TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                target_amount DOUBLE PRECISION NOT NULL,
                current_savings DOUBLE PRECISION NOT NULL,
                monthly_income DOUBLE PRECISION NOT NULL,
                months_remaining INTEGER NOT NULL
            )
        """)


def df():
    with con() as c:
        x = pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY date, id",
            c
        )

    if x.empty:
        return pd.DataFrame(
            columns=["id", "date", "description", "category", "amount", "type"]
        )

    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce")
    x = x.dropna(subset=["date", "amount"])
    x["type"] = x["type"].astype(str).str.lower()
    return x


def expenses():
    return df().query("type == 'expense'").copy()


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "FinSight API running with PostgreSQL"
    }


@app.get("/api/transactions")
def transactions():
    x = df().sort_values(["date", "id"], ascending=False).copy()

    if not x.empty:
        x["date"] = x["date"].dt.strftime("%Y-%m-%d")

    return x.to_dict("records")


@app.post("/api/transactions")
def add(t: Tx):
    if t.amount <= 0 or t.type.lower() not in ["income", "expense"]:
        raise HTTPException(400, "Invalid amount or type")

    try:
        d = pd.to_datetime(t.date).strftime("%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "Invalid date")

    with con() as c:
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO transactions(date, description, category, amount, type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (d, t.description, t.category, t.amount, t.type.lower())
        )
        transaction_id = cur.fetchone()[0]

    return {
        "message": "Transaction added successfully",
        "id": transaction_id
    }


@app.delete("/api/transactions/{id}")
def delete(id: int):
    with con() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM transactions WHERE id = %s", (id,))

        if cur.rowcount == 0:
            raise HTTPException(404, "Not found")

    return {"message": "Deleted"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "CSV required")

    raw = pd.read_csv(io.BytesIO(await file.read()))
    raw.columns = [str(x).strip().lower() for x in raw.columns]

    req = ["date", "description", "category", "amount", "type"]

    if not set(req) <= set(raw.columns):
        raise HTTPException(400, "CSV needs: " + ", ".join(req))

    x = raw[req].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["amount"] = pd.to_numeric(x["amount"], errors="coerce")
    x["type"] = x["type"].astype(str).str.lower()

    x = x.dropna(subset=["date", "amount"])
    x = x[(x["amount"] > 0) & x["type"].isin(["income", "expense"])]

    rows = [
        (
            r.date.strftime("%Y-%m-%d"),
            str(r.description),
            str(r.category),
            float(r.amount),
            r.type
        )
        for _, r in x.iterrows()
    ]

    with con() as c:
        cur = c.cursor()
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO transactions(date, description, category, amount, type)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows
        )

    return {
        "message": "CSV uploaded successfully",
        "rows_imported": len(x)
    }


@app.get("/api/dashboard")
def dashboard():
    x = df()

    if x.empty:
        return {
            "income": 0,
            "expenses": 0,
            "savings": 0,
            "savings_rate": 0,
            "category_data": [],
            "monthly_data": []
        }

    inc = float(x[x["type"] == "income"]["amount"].sum())
    exp = float(x[x["type"] == "expense"]["amount"].sum())
    e = expenses()

    cats = e.groupby("category")["amount"].sum().sort_values(ascending=False)

    x["month"] = x["date"].dt.to_period("M").astype(str)
    m = (
        x.groupby(["month", "type"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    return {
        "income": round(inc, 2),
        "expenses": round(exp, 2),
        "savings": round(inc - exp, 2),
        "savings_rate": round((inc - exp) / inc * 100, 2) if inc else 0,
        "category_data": [
            {"category": k, "amount": round(float(v), 2)}
            for k, v in cats.items()
        ],
        "monthly_data": [
            {
                "month": r["month"],
                "income": float(r.get("income", 0)),
                "expense": float(r.get("expense", 0))
            }
            for _, r in m.iterrows()
        ]
    }


@app.get("/api/statistics")
def statistics():
    x = expenses()["amount"]

    if x.empty:
        return {"message": "No expense data"}

    q1, q3 = x.quantile(.25), x.quantile(.75)

    return {
        "mean": round(float(x.mean()), 2),
        "median": round(float(x.median()), 2),
        "minimum": round(float(x.min()), 2),
        "maximum": round(float(x.max()), 2),
        "std": round(float(x.std()), 2) if len(x) > 1 else 0,
        "variance": round(float(x.var()), 2) if len(x) > 1 else 0,
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(q3 - q1), 2),
        "skewness": round(float(x.skew()), 2) if len(x) > 2 else 0
    }


def forecast_data():
    x = expenses()

    if x.empty:
        return {
            "predicted_expense": 0,
            "historical_average": 0,
            "last_month_expense": 0,
            "method": "No data"
        }

    x["month"] = x["date"].dt.to_period("M").astype(str)
    v = x.groupby("month")["amount"].sum().sort_index().values.astype(float)

    if len(v) < 3:
        p = v.mean()
        method = "Historical Average"
    else:
        slope, intercept = np.polyfit(np.arange(len(v)), v, 1)
        p = .55 * (slope * len(v) + intercept) + .45 * v[-3:].mean()
        method = "Trend Regression + Moving Average"

    return {
        "predicted_expense": round(float(max(p, 0)), 2),
        "historical_average": round(float(v.mean()), 2),
        "last_month_expense": round(float(v[-1]), 2),
        "method": method
    }


@app.get("/api/forecast")
def forecast():
    return forecast_data()


@app.get("/api/overspending")
def overspending():
    x = expenses()

    if x.empty:
        return []

    x["month"] = x["date"].dt.to_period("M").astype(str)

    g = (
        x.groupby(["month", "category"])["amount"]
        .sum()
        .reset_index()
    )

    latest = g["month"].max()
    out = []

    for _, r in g[g["month"] == latest].iterrows():
        h = g[
            (g["category"] == r["category"]) &
            (g["month"] != latest)
        ]["amount"]

        if len(h) and r["amount"] > h.mean() * 1.2:
            extra = float(r["amount"] - h.mean())

            out.append({
                "category": r["category"],
                "current_spending": round(float(r["amount"]), 2),
                "historical_average": round(float(h.mean()), 2),
                "extra_spending": round(extra, 2),
                "increase_percent": round(
                    extra / float(h.mean()) * 100, 2
                )
            })

    return sorted(
        out,
        key=lambda z: z["extra_spending"],
        reverse=True
    )


@app.get("/api/anomalies")
def anomalies():
    x = expenses()

    if len(x) < 8:
        return []

    x = x.copy()

    x["flag"] = IsolationForest(
        contamination=max(.05, min(.15, 2 / len(x))),
        random_state=42
    ).fit_predict(x[["amount"]])

    return [
        {
            "id": int(r["id"]),
            "date": r["date"].strftime("%Y-%m-%d"),
            "description": r["description"],
            "category": r["category"],
            "amount": round(float(r["amount"]), 2)
        }
        for _, r in x[x["flag"] == -1].iterrows()
    ]


def analyze(g):
    f = forecast_data()

    req = max(
        (g.target_amount - g.current_savings) /
        max(g.months_remaining, 1),
        0
    )

    ps = max(g.monthly_income - f["predicted_expense"], 0)
    gap = max(req - ps, 0)

    return {
        "goal_name": g.name,
        "progress_percent": round(
            min(g.current_savings / g.target_amount * 100, 100),
            2
        ) if g.target_amount else 0,
        "required_monthly_savings": round(req, 2),
        "predicted_monthly_expense": f["predicted_expense"],
        "predicted_monthly_savings": round(ps, 2),
        "savings_gap": round(gap, 2),
        "status": "ON TRACK" if ps >= req else "AT RISK"
    }


@app.post("/api/goal")
def goal(g: Goal):
    with con() as c:
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO goals(
                name, target_amount, current_savings,
                monthly_income, months_remaining
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                g.name,
                g.target_amount,
                g.current_savings,
                g.monthly_income,
                g.months_remaining
            )
        )

    return analyze(g)


@app.get("/api/goal")
def get_goal():
    with con() as c:
        cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM goals ORDER BY id DESC LIMIT 1")
        r = cur.fetchone()

    if not r:
        return {"message": "No goal set"}

    return analyze(
        Goal(
            name=r["name"],
            target_amount=r["target_amount"],
            current_savings=r["current_savings"],
            monthly_income=r["monthly_income"],
            months_remaining=r["months_remaining"]
        )
    )


@app.get("/api/recommendations")
def recommendations():
    a = overspending()
    out = []

    for z in a[:5]:
        out.append({
            "category": z["category"],
            "potential_saving": z["extra_spending"],
            "message": (
                f"Reduce {z['category']} toward its historical average "
                f"to save about ₹{z['extra_spending']:,.0f}."
            )
        })

    with con() as c:
        cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM goals ORDER BY id DESC LIMIT 1")
        g = cur.fetchone()

    if g:
        req = max(
            (g["target_amount"] - g["current_savings"]) /
            max(g["months_remaining"], 1),
            0
        )

        ps = max(
            g["monthly_income"] -
            forecast_data()["predicted_expense"],
            0
        )

        if ps < req:
            out.insert(0, {
                "category": "Financial Goal",
                "potential_saving": round(req - ps, 2),
                "message": (
                    f"You need to save ₹{req - ps:,.0f} more per month "
                    f"to stay on track."
                )
            })

    return out or [{
        "category": "General",
        "potential_saving": 0,
        "message": (
            "Your spending is close to historical patterns. "
            "Continue tracking discretionary expenses."
        )
    }]


@app.get("/api/what-if")
def what_if(category: str, reduction_percent: float):
    x = expenses()
    y = x[x["category"].str.lower() == category.lower()]

    if y.empty:
        raise HTTPException(404, "Category not found")

    months = max(x["date"].dt.to_period("M").nunique(), 1)
    avg = float(y["amount"].sum() / months)
    saving = avg * reduction_percent / 100

    return {
        "category": category,
        "average_monthly_category_spending": round(avg, 2),
        "monthly_saving": round(saving, 2),
        "annual_saving": round(saving * 12, 2)
    }
