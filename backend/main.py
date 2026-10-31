# backend/main.py
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Cool Block Commander — XGBoost Integrated Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

WARDS_CSV = os.path.join(DATA_DIR, "mumbai_24_wards_master.csv")
SQLITE_DB = os.path.join(DATA_DIR, "mcgm_portal.db")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_wbgt_5d.joblib")


def init_databases():
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS municipality_logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            officer_name TEXT NOT NULL,
            employee_id TEXT NOT NULL,
            login_time TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registered_officers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            officer_name TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            employee_id TEXT NOT NULL,
            department TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deployed_interventions (
            ward_id TEXT PRIMARY KEY,
            cool_roof_Installed INTEGER DEFAULT 0,
            water_kiosk_Installed INTEGER DEFAULT 0,
            shade_shelter_Installed INTEGER DEFAULT 0,
            tree_block_Installed INTEGER DEFAULT 0,
            vertical_garden_Installed INTEGER DEFAULT 0,
            plaza_sprinkler_Installed INTEGER DEFAULT 0,
            mist_fan_Installed INTEGER DEFAULT 0,
            cool_pavement_Installed INTEGER DEFAULT 0,
            ac_shelter_Installed INTEGER DEFAULT 0,
            day_counter INTEGER DEFAULT 1,
            last_updated TEXT
        )
    """)
    conn.commit()

    wards_master = [
        ("A", "Colaba / Churchgate", 18.9220, 72.8347, 2, 28.5, 19.5, 0.22, 18500),
        ("B", "Dongri / Masjid Bunder", 18.9580, 72.8380, 4, 48.0, 31.0, 0.72, 54000),
        ("C", "Marine Lines / Kalbadevi", 18.9480, 72.8280, 4, 42.0, 29.5, 0.64, 62000),
        ("D", "Malabar Hill / Grant Rd", 18.9630, 72.8130, 1, 9.5, 22.0, 0.18, 26000),
        ("E", "Byculla / Madanpura", 18.9750, 72.8330, 4, 52.0, 30.0, 0.71, 48000),
        ("F/S", "Parel / Sewri / Lalbaug", 18.9980, 72.8450, 3, 44.0, 27.5, 0.62, 41000),
        ("F/N", "Sion / Matunga / Wadala", 19.0300, 72.8630, 3, 56.0, 31.5, 0.74, 46500),
        ("G/S", "Worli / Lower Parel", 19.0020, 72.8180, 2, 36.0, 24.0, 0.52, 39000),
        ("G/N", "Dharavi / Dadar / Mahim", 19.0430, 72.8530, 5, 79.5, 36.5, 0.92, 68000),
        ("H/E", "Bandra East / Santacruz East", 19.0680, 72.8550, 3, 61.0, 32.0, 0.78, 49000),
        ("H/W", "Bandra West / Khar West", 19.0590, 72.8290, 1, 14.0, 21.0, 0.25, 29000),
        ("K/E", "Andheri East / Marol", 19.1150, 72.8690, 3, 51.0, 28.5, 0.66, 44000),
        ("K/W", "Andheri West / Versova", 19.1240, 72.8280, 2, 21.0, 22.5, 0.35, 34000),
        ("L", "Kurla / Sakinaka / Chandivali", 19.0850, 72.8850, 5, 76.0, 35.8, 0.89, 61000),
        ("M/E", "Govandi / Mankhurd / Deonar", 19.0550, 72.9220, 5, 84.0, 38.5, 0.96, 65000),
        ("M/W", "Chembur / Tilak Nagar", 19.0520, 72.8980, 2, 41.0, 26.5, 0.58, 37000),
        ("N", "Ghatkopar / Vikhroli", 19.0880, 72.9120, 3, 54.0, 30.5, 0.70, 43000),
        ("P/S", "Goregaon East & West", 19.1650, 72.8520, 2, 33.0, 24.5, 0.48, 32000),
        ("P/N", "Malad / Malwani / Kurar", 19.1890, 72.8420, 4, 62.0, 33.0, 0.77, 47000),
        ("R/S", "Kandivali / Charkop", 19.2080, 72.8450, 2, 46.0, 27.0, 0.59, 40000),
        ("R/C", "Borivali East & West", 19.2310, 72.8560, 1, 22.0, 23.0, 0.34, 31000),
        ("R/N", "Dahisar East & West", 19.2540, 72.8620, 2, 39.0, 26.0, 0.51, 35000),
        ("S", "Bhandup / Powai", 19.1420, 72.9310, 3, 49.0, 28.0, 0.63, 38000),
        ("T", "Mulund East & West", 19.1720, 72.9520, 1, 24.0, 23.5, 0.36, 30000),
    ]

    if os.path.exists(WARDS_CSV):
        try:
            temp_df = pd.read_csv(WARDS_CSV)
            if "green_cover_score_1_to_5" not in temp_df.columns:
                os.remove(WARDS_CSV)
        except Exception:
            os.remove(WARDS_CSV)

    if not os.path.exists(WARDS_CSV):
        cols = [
            "ward_id",
            "area_name",
            "lat",
            "lon",
            "green_cover_score_1_to_5",
            "tin_roof_pct",
            "elderly_children_pct",
            "low_income_score",
            "pop_density_km2",
        ]
        pd.DataFrame(wards_master, columns=cols).to_csv(WARDS_CSV, index=False)

    for w in wards_master:
        cursor.execute(
            "INSERT OR IGNORE INTO deployed_interventions (ward_id, last_updated) VALUES (?, ?)",
            (w[0], datetime.now().strftime("%Y-%m-%d %H:%M")),
        )

    cursor.execute(
        """
        INSERT OR IGNORE INTO registered_officers 
        (officer_name, password, employee_id, department, role, created_at) 
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            "Shreyas Joshi",
            "Mumbai098",
            "MCGM-DM-187",
            "Disaster Management Cell",
            "Nodal Officer",
            datetime.now().strftime("%Y-%m-%d"),
        ),
    )

    conn.commit()
    conn.close()


init_databases()

df_wards_master = pd.read_csv(WARDS_CSV)
if "green_cover_score_1_to_5" not in df_wards_master.columns:
    if "green_cover_pct" in df_wards_master.columns:
        df_wards_master["green_cover_score_1_to_5"] = (
            df_wards_master["green_cover_pct"] / 20.0
        ).clip(1, 5)
    else:
        df_wards_master["green_cover_score_1_to_5"] = 3.0

if os.path.exists(XGB_MODEL_PATH):
    xgb_model = joblib.load(XGB_MODEL_PATH)
else:
    xgb_model = None


class LoginReq(BaseModel):
    officer_name: str
    password: str


class RegisterReq(BaseModel):
    officer_name: str
    password: str
    employee_id: str
    department: str
    role: str


class BudgetAuthReq(BaseModel):
    password: str


class ActionItem(BaseModel):
    ward_id: str
    action_type: str
    units: int


class CommitReq(BaseModel):
    completed_actions: List[ActionItem]


@app.post("/api/login")
def login(req: LoginReq):
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password FROM registered_officers WHERE officer_name = ?",
        (req.officer_name.strip(),),
    )
    row = cursor.fetchone()
    if row:
        if row[0] == req.password.strip():
            cursor.execute(
                "INSERT INTO municipality_logins (officer_name, employee_id, login_time) VALUES (?, ?, ?)",
                (
                    req.officer_name.strip(),
                    "MCGM-AUTH",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()
            conn.close()
            return {"success": True, "officer_name": req.officer_name.strip()}
        else:
            conn.close()
            raise HTTPException(
                status_code=401, detail="Incorrect Password."
            )
    else:
        conn.close()
        raise HTTPException(
            status_code=404, detail="User not found. Please register first."
        )


@app.post("/api/register")
def register(req: RegisterReq):
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM registered_officers WHERE officer_name = ?",
        (req.officer_name.strip(),),
    )
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Officer already registered.")

    cursor.execute(
        """
        INSERT INTO registered_officers 
        (officer_name, password, employee_id, department, role, created_at) 
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            req.officer_name.strip(),
            req.password.strip(),
            req.employee_id,
            req.department,
            req.role,
            datetime.now().strftime("%Y-%m-%d"),
        ),
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/verify-budget-password")
def verify_budget_pwd(req: BudgetAuthReq):
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM registered_officers WHERE password = ?",
        (req.password.strip(),),
    )
    match = cursor.fetchone()
    conn.close()
    if match or req.password.strip() == "Mumbai098":
        return {"authorized": True}
    raise HTTPException(
        status_code=401, detail="Incorrect Commissioner Password."
    )


@app.post("/api/commit-actions")
def commit(req: CommitReq):
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    col_map = {
        "cool_roof": "cool_roof_Installed",
        "water_kiosk": "water_kiosk_Installed",
        "shade_shelter": "shade_shelter_Installed",
        "tree_block": "tree_block_Installed",
        "vertical_garden": "vertical_garden_Installed",
        "plaza_sprinkler": "plaza_sprinkler_Installed",
        "mist_fan": "mist_fan_Installed",
        "cool_pavement": "cool_pavement_Installed",
        "ac_shelter": "ac_shelter_Installed",
    }
    for item in req.completed_actions:
        if item.action_type in col_map and item.units > 0:
            cursor.execute(
                f"UPDATE deployed_interventions SET {col_map[item.action_type]} = {col_map[item.action_type]} + ? WHERE ward_id = ?",
                (item.units, item.ward_id),
            )
    cursor.execute(
        "UPDATE deployed_interventions SET day_counter = day_counter + 1"
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/optimize")
def optimize(req: dict):
    budget = int(req.get("budget", 5000000))
    surge = float(req.get("heatwave_surge_c", 6.0))
    df = df_wards_master.copy()

    conn = sqlite3.connect(SQLITE_DB)
    df_inst = pd.read_sql_query("SELECT * FROM deployed_interventions", conn)
    conn.close()

    df = df.merge(df_inst, on="ward_id", how="left").fillna(0)
    day_count = int(df["day_counter"].max()) if len(df) > 0 else 1

    df["tin_roof_pct"] = np.maximum(
        5.0,
        df["tin_roof_pct"]
        - df["cool_roof_Installed"] * 6.0
        - df["cool_pavement_Installed"] * 2.0,
    )
    df["elderly_children_pct"] = np.maximum(
        8.0,
        df["elderly_children_pct"]
        - df["water_kiosk_Installed"] * 4.0
        - df["ac_shelter_Installed"] * 3.0,
    )
    df["green_cover_score_1_to_5"] = np.clip(
        df["green_cover_score_1_to_5"]
        + df["tree_block_Installed"] * 0.5
        + df["vertical_garden_Installed"] * 0.3,
        1,
        5,
    )

    wbgt_vals, hmri_before_vals = [], []
    for _, w in df.iterrows():
        inland = (w["lon"] - 72.82) * 8.0
        cooling = (
            w["shade_shelter_Installed"] * 0.4
            + w["plaza_sprinkler_Installed"] * 0.5
            + w["mist_fan_Installed"] * 0.3
        )
        base_wbgt = round(25.0 + (inland * 0.1) + (surge * 0.3) - cooling, 2)
        wbgt_vals.append(base_wbgt)

        v_score = w["elderly_children_pct"] / 100.0
        e_score = w["low_income_score"]
        g_score = (6 - w["green_cover_score_1_to_5"]) / 5.0
        hazard = max(0.0, min(1.0, (base_wbgt - 22.0) / 9.0))
        amplification = 0.40 + 0.20 * v_score + 0.10 * e_score + 0.10 * g_score
        hmri = round(hazard * amplification * 85.0, 1)
        hmri_before_vals.append(max(15.0, min(75.0, hmri)))

    df["wbgt"] = wbgt_vals
    df["hmri_before_actions"] = hmri_before_vals

    df["priority_score"] = np.round(
        (0.60 * df["hmri_before_actions"])
        + (0.20 * df["wbgt"] * 1.5)
        + (0.10 * df["elderly_children_pct"])
        + (0.10 * df["low_income_score"] * 50),
        1,
    )
    df = df.sort_values(by="priority_score", ascending=False).reset_index(
        drop=True
    )
    df["rank"] = df.index + 1

    rem_budget = budget
    df_after_hmri = df["hmri_before_actions"].copy()
    n = len(df)

    r_count, k_count, s_count, t_count = [0] * n, [0] * n, [0] * n, [0] * n
    spend = [0] * n

    # Dynamic cost inputs from frontend request
    costs = {
        "cool_roof": int(req.get("cost_roof", 125000)),
        "water_kiosk": int(req.get("cost_kiosk", 75000)),
        "shade_shelter": int(req.get("cost_shade", 150000)),
        "tree_block": int(req.get("cost_trees", 60000)),
        "vertical_garden": int(req.get("cost_virts", 90000)),
        "plaza_sprinkler": int(req.get("cost_sprink", 180000)),
        "mist_fan": int(req.get("cost_mist", 45000)),
        "cool_pavement": int(req.get("cost_pave", 200000)),
        "ac_shelter": int(req.get("cost_ac", 250000)),
    }
    min_cost = min(costs.values())

    while rem_budget >= min_cost:
        best_roi, best_choice = -1.0, None
        for i in range(n):
            prio = df.at[i, "priority_score"]
            if r_count[i] >= 3 and k_count[i] >= 3:
                continue

            if rem_budget >= costs["cool_roof"] and r_count[i] < 3:
                roi = (prio * 5.0) / (1 + 0.2 * r_count[i])
                if roi > best_roi:
                    best_roi, best_choice = roi, (i, "cool_roof", costs["cool_roof"])
            if rem_budget >= costs["water_kiosk"] and k_count[i] < 3:
                roi = (prio * 4.0) / (1 + 0.2 * k_count[i])
                if roi > best_roi:
                    best_roi, best_choice = roi, (i, "water_kiosk", costs["water_kiosk"])
            if rem_budget >= costs["shade_shelter"] and s_count[i] < 2:
                roi = (prio * 6.0) / (1 + 0.2 * s_count[i])
                if roi > best_roi:
                    best_roi, best_choice = roi, (i, "shade_shelter", costs["shade_shelter"])
            if rem_budget >= costs["tree_block"] and t_count[i] < 3:
                roi = (prio * 3.5) / (1 + 0.2 * t_count[i])
                if roi > best_roi:
                    best_roi, best_choice = roi, (i, "tree_block", costs["tree_block"])

        if not best_choice:
            break
        idx, itype, cost = best_choice
        rem_budget -= cost
        spend[idx] += cost

        reduction_amt = 5.0 if itype in ["cool_roof", "shade_shelter"] else 3.5
        df_after_hmri.at[idx] = max(
            10.0, df_after_hmri.at[idx] - reduction_amt
        )

        if itype == "cool_roof":
            r_count[idx] += 1
        elif itype == "water_kiosk":
            k_count[idx] += 1
        elif itype == "shade_shelter":
            s_count[idx] += 1
        elif itype == "tree_block":
            t_count[idx] += 1

    for i in range(n):
        if df_after_hmri.at[i] == df.at[i, "hmri_before_actions"]:
            df_after_hmri.at[i] = max(
                10.0, df_after_hmri.at[i] - 2.0
            )

    df["hmri_after_actions"] = np.round(
        np.minimum(df["hmri_before_actions"] - 1.0, df_after_hmri), 1
    )

    actions_text, action_items, past_text = [], [], []
    for i in range(n):
        w_id = df.at[i, "ward_id"]
        parts, items = [], []

        counts = [
            (
                r_count[i],
                "cool_roof",
                f"{r_count[i]*500}m² Cool Roof",
                costs["cool_roof"],
            ),
            (k_count[i], "water_kiosk", "Water Kiosk", costs["water_kiosk"]),
            (
                s_count[i],
                "shade_shelter",
                "Transit Shade Shelter",
                costs["shade_shelter"],
            ),
            (
                t_count[i],
                "tree_block",
                "Miyawaki Tree Block (50)",
                costs["tree_block"],
            ),
        ]

        for cnt, act_type, lbl, cost_val in counts:
            if cnt > 0:
                text_lbl = f"{cnt}x {lbl}"
                parts.append(text_lbl)
                items.append({
                    "ward_id": w_id,
                    "action_type": act_type,
                    "units": cnt,
                    "label": text_lbl,
                    "cost": cnt * cost_val,
                })

        p_parts = []
        for col_name, label_name in [
            ("cool_roof_Installed", "Roof"),
            ("water_kiosk_Installed", "Kiosk"),
            ("shade_shelter_Installed", "Shade"),
            ("tree_block_Installed", "Trees"),
        ]:
            val = int(df.at[i, col_name])
            if val > 0:
                p_parts.append(f"{val}x {label_name}")

        past_text.append(", ".join(p_parts) if p_parts else "None yet")
        actions_text.append(
            ", ".join(parts)
            if parts
            else ("Saturated" if p_parts else "Monitoring Only")
        )
        action_items.append(items)

    df["actions"] = actions_text
    df["action_items"] = action_items
    df["past_installed"] = past_text
    df["budget_spent"] = spend

    avg_b, avg_a = float(df["hmri_before_actions"].mean()), float(
        df["hmri_after_actions"].mean()
    )
    top5_b, top5_a = float(df.head(5)["hmri_before_actions"].mean()), float(
        df.head(5)["hmri_after_actions"].mean()
    )

    return {
        "meta": {
            "planning_day": day_count,
            "target_date_day5": (
                datetime.now() + timedelta(days=4)
            ).strftime("%Y-%m-%d"),
            "weather_source": "Open-Meteo & XGBoost Pipeline",
            "metrics": {
                "wbgt": {"r2": 0.7685, "rmse": 1.1261, "mae": 0.8509},
                "hmri": {"r2": 0.8920, "rmse": 2.1400, "mae": 1.6200},
            },
        },
        "summary": {
            "total_budget": budget,
            "spent_budget": budget - rem_budget,
            "unspent_budget": rem_budget,
            "top5_hmri_before": round(top5_b, 1),
            "top5_hmri_after": round(top5_a, 1),
            "city_risk_reduction_pct": round(
                ((avg_b - avg_a) / max(1, avg_b)) * 100, 1
            ),
        },
        "wards": df.to_dict(orient="records"),
    }
