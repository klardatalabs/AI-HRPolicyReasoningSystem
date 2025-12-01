import os
import boto3
import tempfile
import requests
from dotenv import load_dotenv
from datetime import timedelta, datetime
from fastapi import HTTPException
from google.cloud import storage
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from passlib.context import CryptContext
from jose import jwt, JWTError
from monitoring.app_logger import create_logger

logger = create_logger()

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

def get_s3_client():
    """Create and return S3 bucket client + bucket object"""
    try:
        s3 = boto3.client(
            "s3",
            region_name="eu-north-1"
        )
        logger.info("Successfully instantiated S3 client...")
        return s3
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize S3 client: {str(e)}")


def upload_file_to_s3(file_obj, bucket_name: str, key: str):
    """
    Upload a file to S3 using streaming upload.
    Returns a stable internal S3 URI (not public).
    """
    try:
        s3 = get_s3_client()
        logger.info("Received S3 client: ", s3)

        # Ensure file stream is at the beginning
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        logger.info(f"Uploading file instance: {file_obj} to S3...")
        # Streaming upload
        s3.upload_fileobj(
            Fileobj=file_obj,
            Bucket=bucket_name,
            Key=key
        )

        logger.info(f"Successfully uploaded to s3://{bucket_name}/{key}")
        return True, f"s3://{bucket_name}/{key}"

    except Exception as e:
        logger.error(f"S3 upload failed: {str(e)}")
        return False, str(e)


def backup_collection_s3(
        qdrant_client,
        s3_client,
        bucket_name: str,
        collection_name: str,
        qdrant_host: str, key_prefix: str | None = None):
    """
    Create a Qdrant snapshot and upload it to an S3 bucket.
    """
    # Step 1: Create snapshot
    logger.info(f"Creating snapshot for collection: '{collection_name}'...")
    snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)
    snapshot_name = snapshot_info.name

    logger.info(f"Created snapshot: '{snapshot_name}'")

    # Step 2: Build download URL for the snapshot
    download_url = f"{qdrant_host}/collections/{collection_name}/snapshots/{snapshot_name}"

    # Step 3: Download snapshot to a temp file, then upload to S3

    if key_prefix:  # if prefix provided, save to the specific path
        key = f"{key_prefix}/{snapshot_name}"
        logger.info(f"Prefix provided. Saving snapshot to path: '{key}'")
    else:
        # else save it to root of the bucket
        key = f"{snapshot_name}"

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()

        with tempfile.NamedTemporaryFile() as tmp_file:
            # Stream snapshot into local temp file
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)

            tmp_file.flush()
            tmp_file.seek(0)

            # Upload to S3
            try:
                logger.info("Uploading snapshot...")
                s3_client.upload_fileobj(
                    tmp_file,
                    bucket_name,
                    key,
                    ExtraArgs={"ContentType": "application/octet-stream"}
                )
                logger.info("Successfully uploaded snapshot.")
            except Exception as e:
                logger.error("Failed to upload snapshot...")
                logger.info(str(e))
                raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

    return snapshot_name


def restore_latest_snapshot_s3(
        qdrant_client,
        s3_client,
        bucket_name: str,
        collection_name: str,
        key_prefix: str | None = None):
    """
    Restore the latest snapshot for a collection from S3 into Qdrant
    using a presigned URL so Qdrant can download it directly.
    """

    # Ensure prefix is safe for AWS
    prefix = key_prefix or ""   # AWS allows empty prefix
    logger.info(f"[QDRANT RESTORE] Searching for snapshots in s3://{bucket_name}/{prefix}")

    # List snapshots in S3
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
        )
    except Exception as e:
        logger.error(f"[QDRANT RESTORE] Failed to list S3 objects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed listing S3 snapshots: {str(e)}")

    if "Contents" not in response or not response["Contents"]:
        logger.warning(f"[QDRANT RESTORE] No snapshots found under prefix '{prefix}'")
        raise FileNotFoundError("No snapshots found in S3 bucket")

    snapshots = response["Contents"]
    logger.info(f"[QDRANT RESTORE] Found {len(snapshots)} snapshots in S3")

    # Pick latest object by LastModified timestamp
    latest_obj = max(snapshots, key=lambda obj: obj["LastModified"])
    snapshot_key = latest_obj["Key"]   # DO NOT modify — full path already included

    logger.info(f"[QDRANT RESTORE] Latest snapshot selected: {snapshot_key}")

    # Step 3: Generate presigned URL for Qdrant to fetch
    logger.info("[QDRANT RESTORE] Generating presigned S3 URL for snapshot download...")

    try:
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket_name, "Key": snapshot_key},
            ExpiresIn=3600 * 6,  # 6 hours to avoid expiration mid-restore
        )
    except Exception as e:
        logger.error(f"[QDRANT RESTORE] Failed generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed generating presigned URL: {str(e)}")

    logger.info(f"[QDRANT RESTORE] Presigned URL generated.")

    # Optional debug monitoring to ensure presigned URL is correct
    logger.info(f"[QDRANT RESTORE] Presigned URL: {presigned_url}")

    # Step 4: Restore snapshot in Qdrant
    logger.info(f"[QDRANT RESTORE] Beginning restore for collection '{collection_name}'...")
    try:
        qdrant_client.recover_snapshot(
            collection_name=collection_name,
            location=presigned_url,
            wait=True,
            priority="snapshot",
        )
    except Exception as e:
        # collection not found exception
        error_msg = str(e).lower()
        if "not found" in error_msg or "collection" in error_msg:
            logger.error(f"[QDRANT RESTORE] Qdrant reports missing collection '{collection_name}': {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist in Qdrant"
            )
        # any other exception
        logger.exception(f"[QDRANT RESTORE] Failed during Qdrant snapshot restore: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed restoring snapshot: {str(e)}")

    logger.info(
        f"[QDRANT RESTORE] Restore complete: snapshot '{snapshot_key}' applied to '{collection_name}'."
    )
    return snapshot_key


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
                email_id VARCHAR(255) UNIQUE NOT NULL,
                pwd_hash VARCHAR(255) NOT NULL
            );
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

            # 1. Check if user already exists (by email_id)
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

            # 2. Determine if this user should be the first (admin)
            # Count ALL users in the table
            cursor.execute("SELECT COUNT(id) FROM users")
            user_count = cursor.fetchone()[0]

            # If user_count is 0, this is the first user, make them an admin (is_admin=1)
            is_admin_flag = 1 if user_count == 0 else 0

            # 3. Insert the new user
            cursor.execute("""
                INSERT INTO users (email_id, pwd_hash, is_admin)
                VALUES (%s, %s, %s)
            """, (user_obj.email_id, user_obj.hashed_password, is_admin_flag))

            conn.commit()

            # 4. Fetch the last inserted ID
            result = cursor.lastrowid

            if result:
                print("Successfully inserted user with id: ", result)
                return {
                    "success": True,
                    "message": "User successfully created",
                    "user_id": result,
                    "is_admin": is_admin_flag  # Added for clarity
                }
            else:
                print("User insertion completed but no ID returned")
                # This block is unlikely if conn.commit() was successful and an ID column is auto_increment
                return {
                    "success": False,
                    "message": "User insertion completed but no ID returned",
                    "user_id": None,
                    "error_type": "no_id_returned"
                }

        except Exception as e:
            # Ensures any incomplete transaction is cancelled on error
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