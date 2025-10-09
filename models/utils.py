import hashlib
import os
import tempfile
from dotenv import load_dotenv
import requests
from datetime import timedelta, datetime
from fastapi import HTTPException
from google.cloud import storage
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from passlib.context import CryptContext
from jose import jwt, JWTError

from models.data_models import User

load_dotenv()

# MySQL Config
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "tca_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "tca_password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "tca_database")

SECRET_KEY = "a_very_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(
    schemes=["argon2"], deprecated="auto"
)

def backup_collection_gcs(qdrant_client, bucket, collection_name: str, qdrant_host: str):
    """
    Create a Qdrant snapshot and upload it to a GCS bucket.
    """

    # Step 1: Create snapshot
    snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)
    snapshot_name = snapshot_info.name

    # Step 2: Build download URL (local Qdrant)
    download_url = f"{qdrant_host}/collections/{collection_name}/snapshots/{snapshot_name}"

    # Step 3: Download snapshot to a temp file, then upload
    blob = bucket.blob(f"snapshots/{snapshot_name}")

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()

        with tempfile.NamedTemporaryFile() as tmp_file:
            # Stream download into temp file
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)

            tmp_file.flush()
            tmp_file.seek(0)

            # Upload from the temp file
            blob.upload_from_file(
                tmp_file,
                content_type="application/octet-stream"
            )

    return snapshot_name


def restore_latest_snapshot_gcs(qdrant_client, bucket, collection_name: str):
    """Restore the latest snapshot for a collection from GCS into Qdrant using a signed URL."""
    # Find the latest snapshot for this collection
    blobs = list(bucket.list_blobs(prefix="snapshots/"))
    if not blobs:
        raise FileNotFoundError("No snapshots found in GCS bucket")

    # Pick the newest snapshot
    latest_blob = max(blobs, key=lambda b: b.updated)

    # Generate a signed URL for Qdrant to fetch directly
    signed_url = generate_signed_url(
        bucket_name=bucket.name,
        blob_name=latest_blob.name
    )
    # Restore snapshot in Qdrant using the signed URL
    qdrant_client.recover_snapshot(
        collection_name=collection_name,
        location=signed_url,
        wait=True,           # Wait until recovery completes
        priority="snapshot"  # Use snapshot as source of truth
    )
    return latest_blob.name


def generate_signed_url(bucket_name, blob_name, expiration=3600):
    """Generate a signed URL to access a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(
        expiration=timedelta(seconds=expiration),
        method="GET"
    )
    return url

# mysql utils
@contextmanager
def get_db_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        yield connection
    except Error as e:
        print(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if connection and connection.is_connected():
            connection.close()


def init_database():
    """Initialize database and create tables if they don't exist"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'employee',
                    allowed_departments JSON DEFAULT '["finance"]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)

            # Insert default users if they don't exist
            default_users = [
                ("u-employee@company.com", "Default Employee", "employee", '["finance"]'),
                ("u-contractor@company.com", "Default Contractor", "contractor", '[]'),
                ("u-finance@company.com", "Finance Analyst", "finance-analyst", '["finance"]')
            ]

            for email, name, role, departments in default_users:
                cursor.execute("""
                    INSERT IGNORE INTO users (email, name, role, allowed_departments) 
                    VALUES (%s, %s, %s, %s)
                """, (email, name, role, departments))

            conn.commit()
            print("Database initialized successfully")

    except Error as e:
        print(f"Database initialization error: {e}")
        raise

# register user
def insert_user_to_db(user_obj):
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()
            # Check if user already exists
            cursor.execute("""
                SELECT id FROM users WHERE email_id = %s
            """, (user_obj.email_id,))

            existing_user = cursor.fetchone()
            if existing_user:
                return {
                    "success": False,
                    "message": "User already exists",
                    "user_id": existing_user[0],
                    "error_type": "duplicate_email"
                }

            # Insert new user
            cursor.execute("""
                INSERT INTO users (email_id, pwd_hash)
                VALUES (%s, %s)
            """, (user_obj.email_id, user_obj.hashed_password))
            conn.commit()

            result = cursor.lastrowid
            if result:
                print("Successfully inserted user with id: ", result)
                return {
                    "success": True,
                    "message": "User successfully created",
                    "user_id": result
                }
            else:
                print("User insertion completed but no ID returned")
                return {
                    "success": False,
                    "message": "User insertion completed but no ID returned",
                    "user_id": None,
                    "error_type": "no_id_returned"
                }
        except Exception as e:
            conn.rollback()
            print("Error in DB insertion: ", str(e))
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "user_id": None,
                "error_type": "database_error"
            }

def fetch_user_from_db(email: str):
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                select * 
                from users
                where email_id = %s
            """, (email,)
            )
            user = cursor.fetchone()
            if not user:
                return None
            return user
        except Exception as e:
            print("Error fetching user: ", str(e))
            return None


def authenticate_user(email_id, password):
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                select * 
                from users
                where email_id = %s
            """, (email_id,)
            )
            user = cursor.fetchone()
            if not user or not pwd_context.verify(password, user["pwd_hash"]):
                return None
            return user
        except Exception as e:
            print("User authentication error: ", str(e))
            return None


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, SECRET_KEY, algorithm=ALGORITHM
    )
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM]
        )
        username = payload.get("sub")   # username <==> email_id
    except JWTError:
        return None
    if not username:
        print("No username found for this access token...")
        return None
    # get the User object from db using the username
    user_obj = fetch_user_from_db(username)
    return user_obj