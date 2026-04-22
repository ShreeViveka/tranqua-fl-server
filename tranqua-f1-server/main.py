"""
main.py — FastAPI Backend (with per-user data isolation)
=========================================================
Every endpoint now requires user_id so each user
gets their own diary, predictions, and usage data.
"""

import os
import sys
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'collector'))
sys.path.insert(0, os.path.join(ROOT, 'model'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(ROOT, 'data', 'backend.log'), encoding='utf-8'
        )
    ]
)
log = logging.getLogger(__name__)

app = FastAPI(title="Tranqua API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Auth router ───────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.join(ROOT, 'backend'))
    from auth import router as auth_router, init_users_table, init_user_tables
    app.include_router(auth_router)
    init_users_table()
    init_user_tables()
    log.info('[Auth] Auth routes registered.')
except Exception as e:
    log.warning(f'[Auth] Could not register auth: {e}')

# ── Voice router ───────────────────────────────────────────────────────────────
try:
    from voice import router as voice_router
    app.include_router(voice_router)
    log.info('[Voice] Voice routes registered.')
except Exception as e:
    log.warning(f'[Voice] Voice routes not available: {e}')

# ── Predictor (lazy load) ─────────────────────────────────────────────────────
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        from predictor import MentalHealthPredictor
        _predictor = MentalHealthPredictor()
    return _predictor


# ════════════════════════════════════════════════════════════════════════════
# HELPER — get user_id from request header
# ════════════════════════════════════════════════════════════════════════════

def get_user_id(x_user_id: Optional[str] = Header(None)) -> int:
    """
    Every request must include X-User-Id header.
    React sends this automatically after login.
    """
    if not x_user_id:
        raise HTTPException(status_code=401,
            detail="Not authenticated. Please log in.")
    try:
        return int(x_user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid user ID.")


# ════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class DiaryEntryRequest(BaseModel):
    text: str = Field(..., min_length=10)
    date: Optional[str] = None

class PredictionRequest(BaseModel):
    date: Optional[str] = None

class RateContentRequest(BaseModel):
    content_id : int
    was_helpful: bool


# ════════════════════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "running", "app": "Tranqua API", "version": "1.0.0"}

@app.get("/health")
def health():
    status = {"api": "ok", "database": "unknown", "model": "unknown"}
    try:
        from db import get_connection
        conn = get_connection(); conn.close()
        status["database"] = "ok"
    except Exception as e:
        status["database"] = f"error: {e}"
    model_path = os.path.join(ROOT, 'model', 'saved_model.pt')
    if os.path.exists(model_path):
        status["model"] = f"ok ({os.path.getsize(model_path)//1024//1024:.1f} MB)"
    else:
        status["model"] = "not trained yet"
    return status


# ════════════════════════════════════════════════════════════════════════════
# DIARY — filtered by user_id
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/diary")
def save_diary(req: DiaryEntryRequest,
               x_user_id: Optional[str] = Header(None)):
    user_id    = get_user_id(x_user_id)
    entry_date = req.date or str(date.today())
    try:
        datetime.strptime(entry_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(400, "Invalid date. Use YYYY-MM-DD.")

    from db import execute
    execute("""
        INSERT INTO diary_entries (user_id, date, entry_text, word_count)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            entry_text = VALUES(entry_text),
            word_count = VALUES(word_count),
            updated_at = CURRENT_TIMESTAMP
    """, (user_id, entry_date, req.text, len(req.text.split())))

    log.info(f"[API] Diary saved user={user_id} date={entry_date}")
    return {"date": entry_date, "saved": True, "word_count": len(req.text.split())}


@app.get("/api/diary/{entry_date}")
def get_diary(entry_date: str,
              x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    row = execute(
        "SELECT * FROM diary_entries WHERE user_id = %s AND date = %s",
        (user_id, entry_date), fetch='one'
    )
    if not row:
        return {"date": entry_date, "exists": False, "text": "", "word_count": 0}
    return {
        "date"      : entry_date,
        "exists"    : True,
        "text"      : row["entry_text"],
        "word_count": row["word_count"],
    }


@app.get("/api/diary")
def get_recent_diary(days: int = 7,
                     x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    entries = execute("""
        SELECT * FROM diary_entries
        WHERE user_id = %s
        ORDER BY date DESC LIMIT %s
    """, (user_id, days), fetch='all') or []
    return {
        "entries": [{"date": str(e["date"]), "text": e["entry_text"],
                     "word_count": e["word_count"]} for e in entries],
        "count": len(entries)
    }


# ════════════════════════════════════════════════════════════════════════════
# PREDICTION — filtered by user_id
# ════════════════════════════════════════════════════════════════════════════

@app.post("/api/predict")
def predict(req: PredictionRequest,
            x_user_id: Optional[str] = Header(None)):
    user_id    = get_user_id(x_user_id)
    entry_date = req.date or str(date.today())

    from db import execute
    entry = execute(
        "SELECT * FROM diary_entries WHERE user_id = %s AND date = %s",
        (user_id, entry_date), fetch='one'
    )
    if not entry:
        raise HTTPException(404,
            "No diary entry found for this date. Write your diary first.")

    try:
        predictor   = get_predictor()
        target_date = date.fromisoformat(entry_date)
        result      = predictor.predict(
            diary_text = entry["entry_text"],
            target_date= target_date,
            user_id    = user_id,        # pass user_id to predictor
        )
        log.info(f"[API] Prediction user={user_id} date={entry_date}: {result['predicted_state']}")
        return {
            "date"           : result["date"],
            "predicted_state": result["predicted_state"],
            "confidence"     : result["confidence"],
            "emoji"          : result["emoji"],
            "color"          : result["color"],
            "scores"         : result["scores"],
            "score_list"     : result["score_list"],
            "text_weight"    : result["text_weight"],
            "num_weight"     : result["num_weight"],
            "daily_content"  : result["daily_content"],
            "concerns"       : result["concerns"],
            "word_count"     : result["word_count"],
        }
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not ready: {e}")
    except Exception as e:
        log.error(f"[API] Prediction failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/prediction/{entry_date}")
def get_prediction(entry_date: str,
                   x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    from preprocessor import LABEL_COLORS, LABEL_EMOJI
    row = execute(
        "SELECT * FROM predictions WHERE user_id = %s AND date = %s",
        (user_id, entry_date), fetch='one'
    )
    if not row:
        return {"date": entry_date, "exists": False}
    state = row["predicted_state"]
    return {
        "exists"         : True,
        "date"           : str(row["date"]),
        "predicted_state": state,
        "confidence"     : row["confidence"],
        "emoji"          : LABEL_EMOJI.get(state, ""),
        "color"          : LABEL_COLORS.get(state, "#888"),
        "text_weight"    : row["text_weight"],
        "num_weight"     : row.get("numeric_weight", 0.5),
    }


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD — filtered by user_id
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard")
def get_dashboard(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    from predictor import get_daily_content
    from preprocessor import LABEL_COLORS, LABEL_EMOJI

    today = str(date.today())

    diary = execute(
        "SELECT * FROM diary_entries WHERE user_id = %s AND date = %s",
        (user_id, today), fetch='one'
    )
    prediction = execute(
        "SELECT * FROM predictions WHERE user_id = %s AND date = %s",
        (user_id, today), fetch='one'
    )
    summary = execute(
        "SELECT * FROM daily_summary WHERE user_id = %s AND date = %s",
        (user_id, today), fetch='one'
    )

    streak = _calculate_streak(user_id)

    result = {
        "today"     : today,
        "streak"    : streak,
        "has_diary" : diary is not None,
        "diary"     : {"text": diary["entry_text"], "word_count": diary["word_count"]} if diary else None,
        "prediction": None,
        "usage"     : None,
    }

    if prediction:
        state = prediction["predicted_state"]
        result["prediction"] = {
            "predicted_state": state,
            "confidence"     : prediction["confidence"],
            "emoji"          : LABEL_EMOJI.get(state, ""),
            "color"          : LABEL_COLORS.get(state, "#888"),
        }

    if summary:
        result["usage"] = {
            "screen_time_mins" : summary.get("total_screen_time_mins", 0),
            "social_media_mins": summary.get("social_media_mins", 0),
            "work_mins"        : summary.get("work_app_mins", 0),
            "active_mins"      : summary.get("active_time_mins", 0),
            "break_count"      : summary.get("break_count", 0),
            "keystrokes"       : summary.get("keystrokes_count", 0),
        }

    return result


# ════════════════════════════════════════════════════════════════════════════
# TRACKER — filtered by user_id
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/tracker")
def get_tracker(days: int = 7,
                x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    from predictor import generate_weekly_analysis
    from preprocessor import LABEL_COLORS, LABEL_EMOJI

    predictions = execute("""
        SELECT * FROM predictions
        WHERE user_id = %s
        ORDER BY date DESC LIMIT %s
    """, (user_id, days), fetch='all') or []

    summaries = execute("""
        SELECT * FROM daily_summary
        WHERE user_id = %s
        ORDER BY date DESC LIMIT 7
    """, (user_id,), fetch='all') or []

    mood_history = []
    for p in reversed(predictions):
        state = p["predicted_state"]
        mood_history.append({
            "date"           : str(p["date"]),
            "predicted_state": state,
            "confidence"     : p["confidence"],
            "emoji"          : LABEL_EMOJI.get(state, ""),
            "color"          : LABEL_COLORS.get(state, "#888"),
        })

    usage_trend = []
    for s in reversed(summaries):
        usage_trend.append({
            "date"             : str(s["date"]),
            "screen_time_mins" : s.get("total_screen_time_mins", 0) or 0,
            "social_media_mins": s.get("social_media_mins", 0)       or 0,
            "active_mins"      : s.get("active_time_mins", 0)        or 0,
            "late_night_mins"  : s.get("late_night_usage_mins", 0)   or 0,
        })

    weekly = generate_weekly_analysis(predictions, summaries)

    return {
        "mood_history": mood_history,
        "usage_trend" : usage_trend,
        "weekly"      : weekly,
        "days"        : days,
    }


# ════════════════════════════════════════════════════════════════════════════
# USAGE
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/usage/today")
def get_today_usage(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    from categories import get_category_display_name
    from feature_extractor import compute_derived_features

    today = str(date.today())
    apps  = execute("""
        SELECT app_name, category, SUM(duration_secs) AS total_secs
        FROM app_usage WHERE user_id = %s AND date = %s
        GROUP BY app_name, category ORDER BY total_secs DESC
    """, (user_id, today), fetch='all') or []

    summary = execute(
        "SELECT * FROM daily_summary WHERE user_id = %s AND date = %s",
        (user_id, today), fetch='one'
    )

    categories = {}
    for a in apps:
        name = get_category_display_name(a["category"])
        if name not in categories:
            categories[name] = {"mins": 0, "apps": []}
        categories[name]["mins"]  += round(a["total_secs"] / 60, 1)
        categories[name]["apps"].append(a["app_name"])

    derived = compute_derived_features(summary) if summary else {}
    return {
        "date"      : today,
        "categories": categories,
        "top_apps"  : [{"name": a["app_name"], "category": a["category"],
                        "mins": round(a["total_secs"]/60,1)} for a in apps[:5]],
        "summary"   : {
            "total_screen_mins": summary.get("total_screen_time_mins", 0) if summary else 0,
            "active_mins"      : summary.get("active_time_mins", 0)       if summary else 0,
            "idle_mins"        : summary.get("idle_time_mins", 0)         if summary else 0,
            "break_count"      : summary.get("break_count", 0)            if summary else 0,
            "keystrokes"       : summary.get("keystrokes_count", 0)       if summary else 0,
        },
        "insights": derived,
    }


# ════════════════════════════════════════════════════════════════════════════
# CONTENT — filtered by user_id
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/content/today")
def get_today_content(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    from db import execute
    from predictor import get_daily_content

    today = str(date.today())
    row   = execute("""
        SELECT * FROM positive_content
        WHERE user_id = %s AND date = %s
        ORDER BY created_at DESC LIMIT 1
    """, (user_id, today), fetch='one')

    if not row:
        content = get_daily_content("Normal")
        return {"date": today, "exists": False, **content}

    return {
        "date"       : today,
        "exists"     : True,
        "id"         : row["id"],
        "type"       : row["content_type"],
        "text"       : row["content_text"],
        "was_helpful": row["was_helpful"],
    }


@app.post("/api/content/rate")
def rate_content(req: RateContentRequest,
                 x_user_id: Optional[str] = Header(None)):
    get_user_id(x_user_id)   # just verify auth
    from db import rate_positive_content
    rate_positive_content(req.content_id, req.was_helpful)
    return {"rated": True}


# ════════════════════════════════════════════════════════════════════════════
# FL STATUS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/fl/status")
def fl_status(x_user_id: Optional[str] = Header(None)):
    get_user_id(x_user_id)
    from fl_client import (should_upload, get_or_create_client_id,
                           has_enough_local_data, already_uploaded_today)
    can_upload, reason = should_upload()
    return {
        "uploaded_today"   : already_uploaded_today(),
        "can_upload_now"   : can_upload,
        "reason"           : reason,
        "enough_data"      : has_enough_local_data(),
        "client_id_preview": get_or_create_client_id()[:8] + "...",
    }


@app.post("/api/fl/upload")
def trigger_fl_upload(background_tasks: BackgroundTasks,
                      x_user_id: Optional[str] = Header(None)):
    """Manually trigger FL weight upload to Railway server."""
    get_user_id(x_user_id)
    from fl_client import should_upload
    can, reason = should_upload()
    if not can:
        return {"triggered": False, "reason": reason}
    background_tasks.add_task(_do_fl_upload)
    return {"triggered": True, "message": "Upload started in background."}


def _do_fl_upload():
    """Background task: extract weights and upload to Railway FL server."""
    try:
        from fl_client import upload_weight_update
        success = upload_weight_update()
        log.info(f"[FL] Background upload result: {success}")
    except Exception as e:
        log.error(f"[FL] Background upload failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/report")
def download_report(period: str = "week",
                    x_user_id: Optional[str] = Header(None)):
    user_id = get_user_id(x_user_id)
    try:
        sys.path.insert(0, os.path.join(ROOT, 'backend'))
        from report import generate_pdf_report
        pdf_bytes = generate_pdf_report(period, user_id)
        filename  = f"tranqua_report_{period}_{date.today()}.pdf"
        return Response(
            content    = pdf_bytes,
            media_type = "application/pdf",
            headers    = {"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except ImportError:
        raise HTTPException(503, "reportlab not installed. Run: pip install reportlab")
    except Exception as e:
        log.error(f"[API] PDF report failed: {e}")
        raise HTTPException(500, str(e))


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _calculate_streak(user_id: int) -> int:
    from db import execute
    try:
        rows = execute("""
            SELECT date FROM diary_entries
            WHERE user_id = %s
            ORDER BY date DESC LIMIT 30
        """, (user_id,), fetch='all') or []
    except Exception:
        return 0

    if not rows:
        return 0

    streak    = 0
    check_day = date.today()
    for row in rows:
        row_date = row["date"]
        if isinstance(row_date, str):
            row_date = date.fromisoformat(row_date)
        if hasattr(row_date, 'date'):
            row_date = row_date.date()
        if row_date == check_day:
            streak   += 1
            check_day -= timedelta(days=1)
        else:
            break
    return streak


if __name__ == "__main__":
    import uvicorn
    os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1)
