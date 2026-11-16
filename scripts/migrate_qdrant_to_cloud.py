from __future__ import annotations

import os
import sys
import time
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

ENV_FILE = ".env.production"

if not load_dotenv(ENV_FILE):
    print(f"WARNING: Could not load {ENV_FILE}")

print("=" * 70)
print("QDRANT LOCAL -> QDRANT CLOUD MIGRATION")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

LOCAL_QDRANT_PATH = os.getenv(
    "LOCAL_QDRANT_PATH",
    "data/qdrant_storage",
)

CLOUD_URL = os.getenv("QDRANT_URL")
CLOUD_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "financial_rag_analytics",
)

# Start small because your previous request timed out.
# After successful migration, you can increase this.
BATCH_SIZE = int(
    os.getenv(
        "QDRANT_MIGRATION_BATCH_SIZE",
        "5",
    )
)

TIMEOUT = int(
    os.getenv(
        "QDRANT_MIGRATION_TIMEOUT",
        "120",
    )
)

RETRIES = int(
    os.getenv(
        "QDRANT_MIGRATION_RETRIES",
        "3",
    )
)

RETRY_BASE_DELAY = int(
    os.getenv(
        "QDRANT_MIGRATION_RETRY_DELAY",
        "2",
    )
)


# ============================================================
# VECTOR NAME MAPPING
# ============================================================

# Existing local collection
OLD_VECTOR_NAME = "fastembed-BAAI/bge-small-en-v1.5"

# Qdrant Cloud does not allow "/" in vector names.
NEW_VECTOR_NAME = "fastembed-BAAI-bge-small-en-v1-5"


# ============================================================
# VALIDATE CONFIG
# ============================================================

def validate_config() -> None:
    print("\nConfiguration:")
    print(f"  Local Qdrant path : {LOCAL_QDRANT_PATH}")
    print(f"  Cloud URL         : {CLOUD_URL}")
    print(f"  Collection        : {COLLECTION_NAME}")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Timeout           : {TIMEOUT}s")
    print(f"  Retries           : {RETRIES}")
    print(f"  Old vector name   : {OLD_VECTOR_NAME}")
    print(f"  New vector name   : {NEW_VECTOR_NAME}")

    if not CLOUD_URL:
        raise RuntimeError(
            "QDRANT_URL is missing from .env.production"
        )

    if not CLOUD_API_KEY:
        raise RuntimeError(
            "QDRANT_API_KEY is missing from .env.production"
        )

    if BATCH_SIZE < 1:
        raise RuntimeError("BATCH_SIZE must be >= 1")

    if TIMEOUT < 10:
        raise RuntimeError("TIMEOUT should be >= 10 seconds")

    if RETRIES < 1:
        raise RuntimeError("RETRIES must be >= 1")


# ============================================================
# LOCAL CLIENT
# ============================================================

def create_local_client() -> QdrantClient:
    print("\nConnecting to local Qdrant...")

    client = QdrantClient(
        path=LOCAL_QDRANT_PATH,
    )

    print("✓ Local Qdrant connected")

    return client


# ============================================================
# CLOUD CLIENT
# ============================================================

def create_cloud_client() -> QdrantClient:
    print("\nConnecting to Qdrant Cloud...")

    client = QdrantClient(
        url=CLOUD_URL,
        api_key=CLOUD_API_KEY,
        timeout=TIMEOUT,
    )

    print("✓ Qdrant Cloud client created")

    return client


# ============================================================
# GET LOCAL COLLECTION INFORMATION
# ============================================================

def get_local_collection_info(
    local_client: QdrantClient,
) -> Any:

    print("\nReading local collection information...")

    info = local_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("\nLocal collection:")
    print(f"  Name          : {COLLECTION_NAME}")
    print(f"  Status        : {info.status}")
    print(f"  Points        : {info.points_count}")
    print(f"  Vectors       : {info.config.params.vectors}")

    return info


# ============================================================
# EXTRACT VECTOR CONFIG
# ============================================================

def get_vector_config(local_info: Any) -> VectorParams:
    """
    Extract the local dense vector configuration.

    We keep the original vector size and distance.
    Only the vector NAME is changed.
    """

    vectors = local_info.config.params.vectors

    print("\nInspecting local vector configuration...")

    # Named-vector configuration
    if isinstance(vectors, dict):

        if OLD_VECTOR_NAME not in vectors:
            raise RuntimeError(
                f"Expected vector '{OLD_VECTOR_NAME}' "
                f"was not found in local collection.\n"
                f"Available vectors: {list(vectors.keys())}"
            )

        old_config = vectors[OLD_VECTOR_NAME]

        print(f"  Found vector: {OLD_VECTOR_NAME}")
        print(f"  Size       : {old_config.size}")
        print(f"  Distance   : {old_config.distance}")

        return VectorParams(
            size=old_config.size,
            distance=old_config.distance,
        )

    # Single unnamed vector configuration
    print("  Local collection appears to use a single vector.")

    print(f"  Size       : {vectors.size}")
    print(f"  Distance   : {vectors.distance}")

    return VectorParams(
        size=vectors.size,
        distance=vectors.distance,
    )


# ============================================================
# CREATE CLOUD COLLECTION
# ============================================================

def ensure_cloud_collection(
    cloud_client: QdrantClient,
    vector_config: VectorParams,
) -> None:

    print("\nChecking Qdrant Cloud collection...")

    existing_collections = cloud_client.get_collections()

    collection_names = {
        collection.name
        for collection in existing_collections.collections
    }

    if COLLECTION_NAME in collection_names:

        print(
            f"Cloud collection '{COLLECTION_NAME}' already exists."
        )

        info = cloud_client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print(f"Cloud points currently: {info.points_count}")

        print("\nCloud vector configuration:")
        print(f"  {info.config.params.vectors}")

        return

    print(
        f"Creating Cloud collection '{COLLECTION_NAME}'..."
    )

    cloud_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            NEW_VECTOR_NAME: vector_config,
        },
    )

    print("✓ Cloud collection created")

    info = cloud_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    print("\nCreated Cloud collection:")
    print(f"  Points  : {info.points_count}")
    print(f"  Vectors : {info.config.params.vectors}")


# ============================================================
# RENAME VECTOR IN POINT
# ============================================================

def rename_point_vector(
    point: Any,
) -> PointStruct:

    """
    Convert:

        fastembed-BAAI/bge-small-en-v1.5

    into:

        fastembed-BAAI-bge-small-en-v1-5

    while preserving:

        - point ID
        - embedding values
        - payload
    """

    vectors = point.vector

    if vectors is None:
        raise RuntimeError(
            f"Point {point.id} has no vector."
        )

    # --------------------------------------------------------
    # Case 1: Named vector dictionary
    # --------------------------------------------------------

    if isinstance(vectors, dict):

        if OLD_VECTOR_NAME not in vectors:
            raise RuntimeError(
                f"Point {point.id} does not contain "
                f"expected vector '{OLD_VECTOR_NAME}'. "
                f"Available vectors: {list(vectors.keys())}"
            )

        old_vector = vectors[OLD_VECTOR_NAME]

        new_vectors = {
            NEW_VECTOR_NAME: old_vector
        }

    # --------------------------------------------------------
    # Case 2: Unnamed vector
    # --------------------------------------------------------

    else:

        new_vectors = {
            NEW_VECTOR_NAME: vectors
        }

    return PointStruct(
        id=point.id,
        vector=new_vectors,
        payload=point.payload,
    )


# ============================================================
# UPLOAD ONE BATCH WITH RETRIES
# ============================================================

def upload_batch_with_retry(
    cloud_client: QdrantClient,
    points: list[PointStruct],
    batch_number: int,
) -> None:

    for attempt in range(1, RETRIES + 1):

        try:

            print(
                f"\nUploading batch {batch_number} "
                f"({len(points)} points) "
                f"[attempt {attempt}/{RETRIES}]..."
            )

            cloud_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            print(
                f"✓ Batch {batch_number} uploaded successfully"
            )

            return

        except Exception as exc:

            print(
                f"✗ Batch {batch_number} failed"
            )

            print(
                f"  Error type : {type(exc).__name__}"
            )

            print(
                f"  Error      : {exc}"
            )

            if attempt == RETRIES:

                raise RuntimeError(
                    f"Batch {batch_number} failed after "
                    f"{RETRIES} attempts."
                ) from exc

            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))

            print(
                f"  Retrying in {delay} seconds..."
            )

            time.sleep(delay)


# ============================================================
# MIGRATE POINTS
# ============================================================

def migrate_points(
    local_client: QdrantClient,
    cloud_client: QdrantClient,
    total_local_points: int,
) -> None:

    print("\n" + "=" * 70)
    print("STARTING POINT MIGRATION")
    print("=" * 70)

    offset = None
    batch_number = 0
    migrated_points = 0

    while True:

        points, next_offset = local_client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        batch_number += 1

        cloud_points = []

        for point in points:

            converted_point = rename_point_vector(point)

            cloud_points.append(converted_point)

        upload_batch_with_retry(
            cloud_client=cloud_client,
            points=cloud_points,
            batch_number=batch_number,
        )

        migrated_points += len(cloud_points)

        percentage = (
            migrated_points / total_local_points * 100
            if total_local_points > 0
            else 0
        )

        print(
            f"Progress: "
            f"{migrated_points}/{total_local_points} "
            f"({percentage:.2f}%)"
        )

        offset = next_offset

        if offset is None:
            break

    print("\n" + "=" * 70)
    print("POINT MIGRATION FINISHED")
    print("=" * 70)

    print(f"Total migrated: {migrated_points}")


# ============================================================
# VERIFY MIGRATION
# ============================================================

def verify_migration(
    local_client: QdrantClient,
    cloud_client: QdrantClient,
) -> None:

    print("\n" + "=" * 70)
    print("VERIFYING MIGRATION")
    print("=" * 70)

    local_info = local_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    cloud_info = cloud_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    local_count = local_info.points_count
    cloud_count = cloud_info.points_count

    print(f"\nLocal points : {local_count}")
    print(f"Cloud points : {cloud_count}")

    if local_count == cloud_count:

        print(
            "\n✓ SUCCESS: Local and Cloud point counts match."
        )

    else:

        print(
            "\n✗ WARNING: Point counts do not match."
        )

        print(
            f"  Missing/different count: "
            f"{local_count - cloud_count}"
        )

        raise RuntimeError(
            "Migration verification failed because "
            "point counts do not match."
        )

    print("\nCloud collection configuration:")

    print(
        cloud_info.config.params.vectors
    )


# ============================================================
# TEST CLOUD CONNECTION
# ============================================================

def test_cloud_connection(
    cloud_client: QdrantClient,
) -> None:

    print("\nTesting Qdrant Cloud connection...")

    try:

        collections = cloud_client.get_collections()

        print(
            "✓ Cloud connection successful"
        )

        print(
            f"✓ Cloud collections found: "
            f"{len(collections.collections)}"
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not connect to Qdrant Cloud."
        ) from exc


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    validate_config()

    local_client = None
    cloud_client = None

    try:

        # ----------------------------------------------------
        # Create clients
        # ----------------------------------------------------

        local_client = create_local_client()

        cloud_client = create_cloud_client()

        # ----------------------------------------------------
        # Test Cloud connectivity before migration
        # ----------------------------------------------------

        test_cloud_connection(cloud_client)

        # ----------------------------------------------------
        # Read local collection
        # ----------------------------------------------------

        local_info = get_local_collection_info(
            local_client
        )

        total_local_points = local_info.points_count

        if total_local_points == 0:

            print(
                "\nERROR: Local collection contains 0 points."
            )

            sys.exit(1)

        # ----------------------------------------------------
        # Get vector configuration
        # ----------------------------------------------------

        vector_config = get_vector_config(
            local_info
        )

        # ----------------------------------------------------
        # Create / verify Cloud collection
        # ----------------------------------------------------

        ensure_cloud_collection(
            cloud_client=cloud_client,
            vector_config=vector_config,
        )

        # ----------------------------------------------------
        # Migrate points
        # ----------------------------------------------------

        migrate_points(
            local_client=local_client,
            cloud_client=cloud_client,
            total_local_points=total_local_points,
        )

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        verify_migration(
            local_client=local_client,
            cloud_client=cloud_client,
        )

        print("\n" + "=" * 70)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nNext steps:")
        print("1. Update app/config.py VECTOR_NAME")
        print("2. Point API/Admin Qdrant clients to Qdrant Cloud")
        print("3. Test RAG retrieval")
        print("4. Compare local vs Cloud retrieval")
        print("5. Only then remove local Qdrant storage")

    except KeyboardInterrupt:

        print(
            "\n\nMigration interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print("\n" + "=" * 70)
        print("MIGRATION FAILED")
        print("=" * 70)

        print(
            f"\nError type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        print(
            "\nYour local Qdrant data has NOT been modified."
        )

        print(
            "You can safely run the migration again."
        )

        sys.exit(1)

    finally:

        # Explicitly close clients instead of relying on
        # QdrantClient.__del__ during Python shutdown.

        if cloud_client is not None:

            try:
                cloud_client.close()
            except Exception:
                pass

        if local_client is not None:

            try:
                local_client.close()
            except Exception:
                pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()