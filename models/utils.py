import tempfile
import shutil
import requests


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


def restore_collection_gcs(qdrant_client, bucket, snapshot_name: str, collection_name: str):
    """Restore from native snapshot"""
    # Download snapshot from GCS
    blob = bucket.blob(f"snapshots/{snapshot_name}")
    local_path = f"/tmp/{snapshot_name}"
    blob.download_to_filename(local_path)

    # Move to Qdrant snapshots directory
    qdrant_snapshot_path = f"/qdrant/storage/snapshots/{snapshot_name}"
    shutil.move(local_path, qdrant_snapshot_path)

    # Recover from snapshot
    qdrant_client.recover_snapshot(collection_name, qdrant_snapshot_path)
