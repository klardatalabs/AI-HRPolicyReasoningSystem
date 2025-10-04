import tempfile
from dotenv import load_dotenv
import requests
from datetime import timedelta
from google.cloud import storage

load_dotenv()


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