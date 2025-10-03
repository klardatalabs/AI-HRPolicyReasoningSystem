import io
import shutil
import requests
from google.resumable_media.requests import ResumableUpload


def backup_collection_gcs(qdrant_client, bucket, collection_name: str):
    """Backup Qdrant collection into GCS with resumable upload"""
    snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)

    # Find the created snapshot
    snapshots = qdrant_client.list_snapshots(collection_name=collection_name)
    target_snapshot = next((s for s in snapshots if s.name == snapshot_info.name), None)
    if not target_snapshot:
        raise ValueError(f"Snapshot {snapshot_info.name} not found")

    # Snapshot size is needed for resumable upload
    total_size = target_snapshot.size  # Qdrant snapshot usually has this

    # Prepare resumable upload
    blob = bucket.blob(f"snapshots/{snapshot_info.name}")
    transport = requests.Session()

    upload = ResumableUpload(
        upload_url=blob.create_resumable_upload_session(),
        chunk_size=8 * 1024 * 1024,  # 8MB
    )
    upload.initiate(transport,
                    stream=None,
                    metadata={"contentType": "application/octet-stream"},
                    stream_final=False,
                    total_bytes=total_size
                    )

    # Stream snapshot directly into GCS
    with requests.get(target_snapshot.download_url, stream=True, timeout=60) as response:
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue

            stream = io.BytesIO(chunk)
            upload.transmit_next_chunk(transport, stream)

    return snapshot_info.name


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
