# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional

# Optional GenAI client import. Will raise a clear error if used but not installed.
try:
    from google import genai
except Exception:
    genai = None
from passlib.context import CryptContext
from db import get_connection
from fastapi.middleware.cors import CORSMiddleware 
import pymysql
from fastapi import UploadFile, File
import tempfile
import shutil
from stutter_inference import StutterPredictor
from stutter_model import ConvLSTM_Stutter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    # Truncate the password to 72 characters
    return pwd_context.hash(password[:72])

def verify_password(plain_password: str, password: str) -> bool:
    return plain_password == password

# Pydantic models for request validation
class User(BaseModel):
    email: str
    password: str


class GenAIRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-2.5-flash"

@app.post("/register")
def register_user(user: User):
    conn = get_connection()
    cursor = conn.cursor()

   
    cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail="User already exists")

    cursor.execute("INSERT INTO nyu_hack_2025.users (email, password) VALUES (%s, %s)", (user.email, user.password))
    conn.commit()

    cursor.close()
    conn.close()
    return {"message": "User registered successfully"}

@app.post("/login")
def login_user(user: User):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM nyu_hack_2025.users WHERE email = %s", (user.email))
    result = cursor.fetchone()

    if not result or not verify_password(user.password, result[1]):
        cursor.close()
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = result[0]

    cursor.close()
    conn.close()
    return {"id": user_id}

@app.get("/api/metrics/progress-over-time/{user_id}", summary="2. Progress over the days (line chart)")
def get_progress_over_time(user_id: int):
    """
    Fetches the total stuttering events for all days to show progress.
    Returns data formatted for a line chart.
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    try:
        # Get all records, ordered by date
        sql = """
            SELECT record_date, (class1 + class2 + class3 + class4 + class5) as total_events
            FROM classes 
            WHERE id = %s 
            ORDER BY record_date ASC
        """
        cursor.execute(sql, (user_id,))
        results = cursor.fetchall()
        
        if not results:
            raise HTTPException(status_code=404, detail="No data found for this user.")

        # Format for React chart libraries
        return {
            "labels": [r['record_date'].strftime('%Y-%m-%d') for r in results], # Dates
            "datasets": [{
                "label": "Total Stuttering Events",
                "data": [r['total_events'] for r in results], # Counts
            }]
        }
    
    finally:
        cursor.close()
        conn.close()


@app.get("/api/metrics/fluency-score/{user_id}", summary="4. Fluency Score over time (line chart)")
def get_fluency_score_over_time(user_id: int):
    """
    Fetches the fluency score (total_events / total_seconds) for all days.
    Returns data formatted for a line chart.
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    try:

        sql = """
            SELECT 
                record_date, 
                ((class1 + class2 + class3 + class4 + class5) / total_seconds) as fluency_score
            FROM classes 
            WHERE 
                id = %s AND
                total_seconds IS NOT NULL AND
                total_seconds > 0
            ORDER BY record_date ASC
        """
        cursor.execute(sql, (user_id,))
        results = cursor.fetchall()
        
        if not results:
            return {"labels": [], "datasets": []}

        return {
            "labels": [r['record_date'].strftime('%Y-%m-%d') for r in results], # Dates
            "datasets": [{
                "label": "Fluency Score",
                "data": [r['fluency_score'] for r in results], # Scores
            }]
        }
    
    finally:
        cursor.close()
        conn.close()

@app.get("/api/metrics/classwise-stuttering-latest/{user_id}", summary="Class-wise stuttering for most recent day")
def get_classwise_stuttering_latest(user_id: int):
    """
    Fetches the class-wise stuttering counts for the *most recent* recorded date.
    Ideal for a bar or donut chart showing the last session's breakdown.
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    
    try:
        sql = """
            SELECT 
                record_date,
                class1,
                class2,
                class3,
                class4,
                class5
            FROM classes 
            WHERE id = %s 
            ORDER BY record_date DESC 
            LIMIT 1
        """
        cursor.execute(sql, (user_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="No data found for this user.")
            
        # Format for React chart libraries
        return {
            "record_date": result['record_date'].strftime('%Y-%m-%d'),
            "labels": ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"],
            "datasets": [{
                "label": "Events from last session",
                "data": [
                    result['class1'],
                    result['class2'],
                    result['class3'],
                    result['class4'],
                    result['class5']
                ],
            }]
        }
    
    finally:
        cursor.close()
        conn.close()


@app.api_route("/api/genai/generate", methods=["GET", "POST"], summary="Generate text via Google GenAI")
def generate_with_genai(req: Optional[GenAIRequest] = None, prompt: Optional[str] = None, model: Optional[str] = None):
    
    # Determine prompt and model from body (req) or query params
    prompt_text = "Give me a approx 100-word paragraph taken from any novel. Do not add explanations, disclaimers, or extra text. Output only the paragraph."
    model_name = "gemini-2.5-flash"

    api_key = "AIzaSyCh6LATE46apIwZM6Xcp0AOhmlPohusxAo"
    if not api_key:
        raise HTTPException(status_code=500, detail="GENAI_API_KEY not configured in environment")

    if genai is None:
        raise HTTPException(status_code=500, detail="google.genai library not available on server")

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GenAI request failed: {str(e)}")

    return {"text": getattr(response, "text", str(response))}

@app.post("/api/stutter/predict", summary="Predict stuttering from uploaded WAV file")
def predict_stutter():
    try:
        # Always use your local test file
        test_path = "sample.wav"

        predictor = StutterPredictor(
            model_class=ConvLSTM_Stutter,
            checkpoint_path="best_model-2.pt",
            device="cpu"
        )
        probs, preds = predictor.predict(test_path)
        print("probs ", probs)
        print("preds ", preds)
        return {"array":test_path,"probabilities": probs.tolist(), "predictions": preds.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")