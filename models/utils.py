import shutil


def backup_collection_gcs(qdrant_client, bucket, collection_name: str):
    """Native snapshot into GCS bucket"""
    # Create snapshot directly on Qdrant server
    snapshot_info = qdrant_client.create_snapshot(collection_name=collection_name)

    # Upload snapshot file directly to GCS
    snapshot_path = f"/qdrant/storage/snapshots/{snapshot_info.name}"
    blob = bucket.blob(f"snapshots/{snapshot_info.name}")
    blob.upload_from_filename(snapshot_path)

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