# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from db import get_connection

app = FastAPI()

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

    # Fetch the user from the database
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


# @app.get("/users")
# def get_users():
#     conn = get_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, email FROM nyu_hack_2025.users")
#     users = cursor.fetchall()

#     cursor.close()
#     conn.close()
#     return {"users": users}
