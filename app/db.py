from google.cloud.sql.connector import Connector
import pymysql

connector = Connector()

def get_connection():
    conn = connector.connect(
        "nyu-hackathon-backend:us-central1:nyu-hackathon-db",  
        "pymysql",
        user="db_user",               
        password="NYUhackathon@2025",
        db="nyu_hack_2025"               
    )
    return conn
