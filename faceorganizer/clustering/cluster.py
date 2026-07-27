"""DBSCAN-based face clustering."""

from __future__ import annotations

import math
import sqlite3

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, MIN_CLUSTER_SIZE
from faceorganizer.database.core import (
    clear_clusters,
    get_cluster_by_id,
    get_cluster_centroids,
    get_cluster_embeddings,
    get_embeddings_by_ids,
    get_named_clusters,
    get_unassigned_embeddings,
    insert_cluster,
    merge_clusters,
    update_face_clusters_batch,
)
from faceorganizer.logging_config import get_logger

from .embeddings import load_normalized_embeddings

log = get_logger("clustering.cluster")


def run_clustering(
    conn: sqlite3.Connection,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
    min_samples: int = MIN_CLUSTER_SIZE,
) -> int:
    """Cluster all face embeddings and write results to the database.

    Clears any existing cluster assignments before running.
    Returns the number of clusters created.
    """
    face_ids, embeddings = load_normalized_embeddings(conn)
    if len(face_ids) == 0:
        log.warning("No faces to cluster")
        return 0

    # Embeddings are already L2-normalised, so cosine distance equals
    # half the squared Euclidean distance: cos_dist = ||a-b||² / 2.
    # Switching to metric='euclidean' + algorithm='ball_tree' changes
    # neighbourhood search from O(N²) brute-force to O(N log N), which
    # is critical for libraries of 100 K+ faces.
    # Equivalent eps: euclidean_eps = sqrt(2 * cosine_eps)
    eps_euclidean = math.sqrt(2.0 * eps)
    log.info(
        "Running DBSCAN (cosine eps=%.3f → euclidean eps=%.3f, min_samples=%d) on %d faces",
        eps, eps_euclidean, min_samples, len(face_ids),
    )
    dbscan = DBSCAN(
        eps=eps_euclidean,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    labels = dbscan.fit_predict(embeddings)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 means noise / unclustered
    num_clusters = len(unique_labels)
    noise_count = int(np.sum(labels == -1))

    log.info("Found %d clusters, %d unclustered faces", num_clusters, noise_count)

    # Clear old clusters and write new ones
    clear_clusters(conn)

    if num_clusters == 0:
        return 0

    # Create cluster records with auto-generated names and map DBSCAN label -> DB id
    label_to_cluster_id: dict[int, int] = {}
    for label in sorted(unique_labels):
        name = f"Person_{label + 1:03d}"
        cluster_id = insert_cluster(conn, name)
        label_to_cluster_id[label] = cluster_id

    # Build batch assignments: (cluster_id, face_id)
    assignments: list[tuple[int | None, int]] = []
    for face_id, label in zip(face_ids, labels, strict=True):
        cluster_id = label_to_cluster_id.get(int(label))  # None for noise
        assignments.append((cluster_id, face_id))

    update_face_clusters_batch(conn, assignments)
    log.info("Wrote %d cluster assignments to database", len(assignments))

    conn.execute("PRAGMA optimize")
    return num_clusters


def run_incremental_clustering(
    conn: sqlite3.Connection,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
    min_samples: int = MIN_CLUSTER_SIZE,
    maybe_eps: float | None = None,
) -> dict[str, int]:
    """Cluster only unassigned faces, preserving existing clusters.

    1. Compute centroids for each existing cluster.
    2. For each unassigned face, assign it to the nearest cluster if the
       cosine distance is within *eps*.
    3. Run DBSCAN on any remaining unassigned faces to discover new clusters.
    4. If *maybe_eps* is set, new clusters whose centroid is within that
       distance of a user-named cluster get labeled ``maybe <name>``.

    Returns a dict with keys: assigned (matched to existing), new_clusters,
    maybe_clusters, new_noise (still unassigned after DBSCAN).
    """
    # Load unassigned face embeddings
    face_ids, embeddings = get_unassigned_embeddings(conn)
    if len(face_ids) == 0:
        log.info("No unassigned faces — nothing to do")
        return {"assigned": 0, "new_clusters": 0, "maybe_clusters": 0, "new_noise": 0}

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    log.info("Incremental clustering: %d unassigned faces", len(face_ids))

    # Step 1 — match against existing cluster centroids
    centroids = get_cluster_centroids(conn)
    assignments: list[tuple[int | None, int]] = []
    remaining_idx: list[int] = []  # indices into face_ids/embeddings
    assigned_count = 0

    if centroids:
        cids = list(centroids.keys())
        centroid_matrix = np.array([centroids[c] for c in cids], dtype=np.float32)
        # Cosine distance = 1 - dot product (vectors are unit-length)
        dots = embeddings @ centroid_matrix.T  # (N, C)
        best_idx = np.argmax(dots, axis=1)
        best_dist = 1.0 - dots[np.arange(len(dots)), best_idx]

        for i, (fid, dist) in enumerate(zip(face_ids, best_dist, strict=True)):
            if dist <= eps:
                assignments.append((cids[best_idx[i]], fid))
                assigned_count += 1
            else:
                remaining_idx.append(i)
    else:
        remaining_idx = list(range(len(face_ids)))

    log.info("Matched %d faces to existing clusters", assigned_count)

    # Step 2 — DBSCAN on the leftovers
    new_clusters = 0
    maybe_count = 0
    new_noise = 0

    if remaining_idx:
        rem_ids = [face_ids[i] for i in remaining_idx]
        rem_emb = embeddings[remaining_idx]

        eps_euclidean = math.sqrt(2.0 * eps)
        log.info(
            "Running DBSCAN on %d remaining faces "
            "(cosine eps=%.3f → euclidean eps=%.3f, min_samples=%d)",
            len(rem_ids), eps, eps_euclidean, min_samples,
        )
        dbscan = DBSCAN(
            eps=eps_euclidean,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="ball_tree",
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(rem_emb)

        unique_labels = set(labels)
        unique_labels.discard(-1)
        new_clusters = len(unique_labels)
        new_noise = int(np.sum(labels == -1))

        # Pre-compute named cluster centroids for "maybe" matching
        named_centroid_data = None
        if maybe_eps is not None:
            named_clusters = get_named_clusters(conn)
            if named_clusters:
                all_centroids = get_cluster_centroids(conn)
                named_cids = [cid for cid in named_clusters if cid in all_centroids]
                if named_cids:
                    named_vecs = np.array(
                        [all_centroids[cid] for cid in named_cids], dtype=np.float32
                    )
                    named_centroid_data = (named_cids, named_vecs, named_clusters)

        # Create new cluster records, applying "maybe" names where appropriate
        label_to_cluster_id: dict[int, int] = {}
        for label in sorted(unique_labels):
            name = None

            if named_centroid_data is not None:
                n_cids, n_matrix, n_names = named_centroid_data
                # Compute centroid of this new DBSCAN cluster
                mask = labels == label
                cluster_embs = rem_emb[mask]
                centroid = cluster_embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    centroid = centroid / norm
                dots = centroid @ n_matrix.T
                best_i = int(np.argmax(dots))
                best_dist = 1.0 - dots[best_i]
                if best_dist <= maybe_eps:
                    matched_name = n_names[n_cids[best_i]]
                    name = f"maybe {matched_name}"
                    maybe_count += 1
                    log.info(
                        "New cluster matches named cluster '%s' (dist=%.3f) → '%s'",
                        matched_name, best_dist, name,
                    )

            if name is None:
                name = _next_auto_name(conn)

            cluster_id = insert_cluster(conn, name)
            label_to_cluster_id[label] = cluster_id

        for fid, label in zip(rem_ids, labels, strict=True):
            cluster_id = label_to_cluster_id.get(int(label))
            assignments.append((cluster_id, fid))

    if assignments:
        update_face_clusters_batch(conn, assignments)

    conn.execute("PRAGMA optimize")
    log.info(
        "Incremental result: %d assigned to existing, %d new clusters "
        "(%d maybe), %d noise",
        assigned_count, new_clusters, maybe_count, new_noise,
    )
    return {
        "assigned": assigned_count,
        "new_clusters": new_clusters,
        "maybe_clusters": maybe_count,
        "new_noise": new_noise,
    }


def run_recluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
    min_samples: int = MIN_CLUSTER_SIZE,
) -> dict[str, int]:
    """Re-cluster a single cluster's faces using Agglomerative Clustering.

    Uses average-linkage agglomerative clustering to avoid DBSCAN's chaining
    effect, where long transitive chains connect dissimilar faces.  Average
    linkage requires the *mean* pairwise distance within a cluster to stay
    below the threshold, producing tighter, more meaningful sub-clusters.

    Faces in sub-clusters smaller than *min_samples* are treated as noise
    (unassigned).  The largest sub-cluster keeps the original cluster name;
    smaller ones get auto-generated names.

    Returns a dict with keys: original_faces, new_clusters, noise.
    """
    cluster = get_cluster_by_id(conn, cluster_id)
    if cluster is None:
        raise ValueError(f"Cluster {cluster_id} not found")

    face_ids, embeddings = get_cluster_embeddings(conn, cluster_id)
    if len(face_ids) == 0:
        log.warning("Cluster %d has no faces to recluster", cluster_id)
        return {"original_faces": 0, "new_clusters": 0, "noise": 0}

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    log.info(
        "Reclustering cluster %d ('%s', %d faces) with eps=%.3f, min_samples=%d "
        "(agglomerative average-linkage)",
        cluster_id, cluster.name, len(face_ids), eps, min_samples,
    )

    # Agglomerative clustering with average linkage and cosine distance.
    # distance_threshold=eps means clusters are merged only while their
    # average inter-point cosine distance stays below eps.
    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=eps,
        metric="cosine",
        linkage="average",
    )
    labels = agg.fit_predict(embeddings)

    unique_labels = set(labels)
    num_raw_clusters = len(unique_labels)

    # Treat sub-clusters smaller than min_samples as noise (label → -1)
    label_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
    for lbl, count in label_counts.items():
        if count < min_samples:
            labels[labels == lbl] = -1

    unique_labels = set(labels)
    unique_labels.discard(-1)
    num_clusters = len(unique_labels)
    noise_count = int(np.sum(labels == -1))

    log.info(
        "Recluster found %d sub-clusters (%d before min_samples filter), %d noise faces",
        num_clusters, num_raw_clusters, noise_count,
    )

    if num_clusters <= 1 and noise_count == 0:
        log.info("Cluster is already cohesive — no changes made")
        return {"original_faces": len(face_ids), "new_clusters": 0, "noise": 0}

    # Find the largest sub-cluster — it keeps the original cluster name/id
    label_counts = {lbl: int(np.sum(labels == lbl)) for lbl in unique_labels}
    largest_label = max(label_counts, key=label_counts.get) if label_counts else None

    assignments: list[tuple[int | None, int]] = []
    new_cluster_labels: dict[int, int] = {}  # agg label -> new DB cluster id

    for label in sorted(unique_labels):
        if label == largest_label:
            new_cluster_labels[label] = cluster_id
        else:
            name = _next_auto_name(conn)
            new_id = insert_cluster(conn, name)
            new_cluster_labels[label] = new_id

    for face_id, label in zip(face_ids, labels, strict=True):
        cid = new_cluster_labels.get(int(label))  # None for noise
        assignments.append((cid, face_id))

    update_face_clusters_batch(conn, assignments)
    conn.execute("PRAGMA optimize")

    # Count new clusters created (excluding the original)
    new_clusters_created = num_clusters - 1 if largest_label is not None else num_clusters

    log.info(
        "Recluster complete: kept %d faces in '%s', created %d new clusters, %d noise",
        label_counts.get(largest_label, 0) if largest_label is not None else 0,
        cluster.name,
        new_clusters_created,
        noise_count,
    )
    return {
        "original_faces": len(face_ids),
        "new_clusters": new_clusters_created,
        "noise": noise_count,
    }


def split_faces_by_similarity(
    conn: sqlite3.Connection,
    face_ids: list[int],
    base_name: str,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[int]:
    """Split selected faces into similarity-grouped clusters.

    Loads embeddings for *face_ids*, runs agglomerative average-linkage
    clustering with *eps* cosine-distance threshold, then creates one new
    cluster per group.  All faces are assigned (no noise/unassigned) because
    the caller has explicitly chosen them.

    Returns the list of newly created cluster IDs.
    """
    if not face_ids:
        return []

    out_ids, embeddings = get_embeddings_by_ids(conn, face_ids)
    if not out_ids:
        return []

    # L2-normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    if len(out_ids) == 1:
        new_id = insert_cluster(conn, base_name)
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (new_id, out_ids[0]))
        conn.commit()
        return [new_id]

    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=eps,
        metric="cosine",
        linkage="average",
    )
    labels = agg.fit_predict(embeddings)

    unique_labels = sorted(set(labels))
    new_cluster_ids: list[int] = []

    for i, lbl in enumerate(unique_labels):
        group = [out_ids[j] for j in range(len(out_ids)) if labels[j] == lbl]
        cluster_name = f"{base_name} {i + 1}" if len(unique_labels) > 1 else base_name
        new_id = insert_cluster(conn, cluster_name)
        placeholders = ",".join("?" * len(group))
        conn.execute(
            f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})",
            [new_id, *group],
        )
        new_cluster_ids.append(new_id)

    conn.commit()
    log.info(
        "split_faces_by_similarity: %d faces → %d cluster(s) (eps=%.3f)",
        len(out_ids), len(new_cluster_ids), eps,
    )
    return new_cluster_ids


# Tracks the highest Person_NNN number handed out this process, so repeated
# calls within one clustering pass (before earlier inserts are visible to a
# fresh MAX() query) don't collide. Deliberately module-level rather than a
# mutable default argument, to avoid the classic shared-default-value gotcha.
_next_auto_name_cache: dict[str, int] = {}


def _next_auto_name(conn: sqlite3.Connection) -> str:
    """Generate the next Person_NNN name that doesn't conflict with existing clusters."""
    # Find the current max N among Person_NNN names
    cur = conn.execute(
        """SELECT MAX(CAST(SUBSTR(name, 8) AS INTEGER))
           FROM clusters
           WHERE name GLOB 'Person_[0-9][0-9][0-9]*'"""
    )
    row = cur.fetchone()
    max_n = row[0] if row and row[0] is not None else 0

    last_given = _next_auto_name_cache.get("last", 0)
    n = max(max_n, last_given) + 1
    _next_auto_name_cache["last"] = n
    return f"Person_{n:03d}"


def absorb_after_rename(
    conn: sqlite3.Connection,
    cluster_id: int,
    confirmed_name: str,
    eps: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, int]:
    """After a cluster is given a confirmed user name, learn from it immediately.

    1. Any unassigned faces within *eps* cosine distance of the cluster centroid
       are pulled into the cluster.
    2. Any existing ``maybe <confirmed_name>`` clusters are merged in.

    Returns a dict with keys: absorbed_faces, merged_clusters.
    """
    centroids = get_cluster_centroids(conn)
    centroid = centroids.get(cluster_id)

    absorbed_faces = 0
    if centroid is not None:
        face_ids, embeddings = get_unassigned_embeddings(conn)
        if face_ids:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms

            dots = embeddings @ centroid  # (N,)
            dists = 1.0 - dots

            to_absorb = [
                (cluster_id, fid)
                for fid, dist in zip(face_ids, dists, strict=True)
                if dist <= eps
            ]
            if to_absorb:
                update_face_clusters_batch(conn, to_absorb)
                absorbed_faces = len(to_absorb)

    # Merge any "maybe <name>" clusters into the confirmed one
    maybe_name = f"maybe {confirmed_name}"
    cur = conn.execute(
        "SELECT id FROM clusters WHERE name = ? AND id != ?",
        (maybe_name, cluster_id),
    )
    maybe_ids = [r[0] for r in cur.fetchall()]
    for mid in maybe_ids:
        merge_clusters(conn, cluster_id, mid)

    merged_clusters = len(maybe_ids)
    log.info(
        "absorb_after_rename '%s' (cluster %d): absorbed %d unassigned face(s), "
        "merged %d maybe-cluster(s)",
        confirmed_name, cluster_id, absorbed_faces, merged_clusters,
    )
    return {"absorbed_faces": absorbed_faces, "merged_clusters": merged_clusters}
