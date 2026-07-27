"""Flask web application for reviewing and managing face clusters."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, render_template, request, send_file

from faceorganizer import actions
from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD, get_data_dir, get_db_path
from faceorganizer.database.core import (
    get_cluster_by_id,
    get_clusters,
    get_dismissed_faces,
    get_face_by_id,
    get_faces_for_cluster,
    get_photos_for_cluster,
    get_scan_stats,
    update_face_cluster,
    update_face_clusters_batch,
)
from faceorganizer.database.schema import configure_connection, init_db
from faceorganizer.logging_config import get_logger
from faceorganizer.web.settings import Settings
from faceorganizer.web.tasks import create_task, get_task, run_in_background
from faceorganizer.web.thumbnails import get_or_create_thumbnail

log = get_logger("web.app")


def _require_int(value, name: str) -> int:
    """Validate and convert a value to int, raising ValueError on failure."""
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer") from e


def create_app(scan_root: Path) -> Flask:
    """Create and configure the Flask app for a given scan root."""
    log.info("Starting web app for scan root: %s", scan_root)
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["SCAN_ROOT"] = scan_root
    db_path = get_db_path(scan_root)
    app.config["DB_PATH"] = db_path

    # Initialize the schema once at startup, not per-request
    init_db(db_path).close()

    # Load persistent settings
    data_dir = get_data_dir(scan_root)
    app.config["SETTINGS"] = Settings.load(data_dir)

    def get_conn() -> sqlite3.Connection:
        """Get or create a per-request database connection."""
        if "db_conn" not in g:
            conn = sqlite3.connect(str(db_path))
            configure_connection(conn)
            g.db_conn = conn
        return g.db_conn

    @app.teardown_appcontext
    def close_conn(exc):
        """Close the per-request database connection."""
        conn = g.pop("db_conn", None)
        if conn is not None:
            conn.close()

    # ── Pages ──────────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        conn = get_conn()
        stats = get_scan_stats(conn)
        clusters = get_clusters(conn)

        # Fetch representative face (highest confidence) per cluster
        if clusters:
            cluster_ids = [c.id for c in clusters]
            placeholders = ",".join("?" * len(cluster_ids))
            cur = conn.execute(
                f"""SELECT cluster_id, face_id, photo_id, path,
                           bbox_x, bbox_y, bbox_w, bbox_h, confidence
                    FROM (
                        SELECT f.cluster_id, f.id AS face_id, f.photo_id, p.path,
                               f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                               f.detection_confidence AS confidence,
                               ROW_NUMBER() OVER (
                                   PARTITION BY f.cluster_id
                                   ORDER BY f.detection_confidence DESC
                               ) AS rn
                        FROM faces f
                        JOIN photos p ON p.id = f.photo_id
                        WHERE f.cluster_id IN ({placeholders})
                          AND f.dismissed = 0
                    ) WHERE rn = 1""",
                cluster_ids,
            )
            rep_faces = {}
            for r in cur:
                rep_faces[r[0]] = {
                    "face_id": r[1], "photo_id": r[2], "photo_path": r[3],
                    "bbox_x": r[4], "bbox_y": r[5], "bbox_w": r[6], "bbox_h": r[7],
                    "confidence": r[8],
                }
            for c in clusters:
                c._rep_face = rep_faces.get(c.id)

        return render_template(
            "dashboard.html", stats=stats, clusters=clusters,
            scan_root=str(scan_root), active_page="dashboard",
        )

    @app.route("/person/<int:cluster_id>")
    def person_detail(cluster_id):
        conn = get_conn()
        cluster = get_cluster_by_id(conn, cluster_id)
        if cluster is None:
            abort(404)
        faces = get_faces_for_cluster(conn, cluster_id)
        photos = get_photos_for_cluster(conn, cluster_id)
        all_clusters = get_clusters(conn)

        # Fetch representative face id per cluster for the drop sidebar
        if all_clusters:
            other_ids = [c.id for c in all_clusters if c.id != cluster_id]
            if other_ids:
                placeholders = ",".join("?" * len(other_ids))
                cur = conn.execute(
                    f"""SELECT cluster_id, face_id FROM (
                            SELECT f.cluster_id, f.id AS face_id,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY f.cluster_id
                                       ORDER BY f.detection_confidence DESC
                                   ) AS rn
                            FROM faces f WHERE f.cluster_id IN ({placeholders})
                              AND f.dismissed = 0
                        ) WHERE rn = 1""",
                    other_ids,
                )
                rep_map = {r[0]: r[1] for r in cur}
                for c in all_clusters:
                    c._rep_face_id = rep_map.get(c.id)

        return render_template(
            "person.html", cluster=cluster, faces=faces,
            photos_count=len(photos), all_clusters=all_clusters,
            active_page="person", default_threshold=DEFAULT_CLUSTER_THRESHOLD,
        )

    @app.route("/review")
    def review():
        conn = get_conn()
        clusters = get_clusters(conn)

        # Fetch top 6 faces per cluster using window function
        cluster_data = []
        if clusters:
            cluster_ids = [c.id for c in clusters]
            placeholders = ",".join("?" * len(cluster_ids))
            cur = conn.execute(
                f"""SELECT cluster_id, face_id, photo_id, path,
                           bbox_x, bbox_y, bbox_w, bbox_h, confidence
                    FROM (
                        SELECT f.cluster_id, f.id AS face_id, f.photo_id, p.path,
                               f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                               f.detection_confidence AS confidence,
                               ROW_NUMBER() OVER (
                                   PARTITION BY f.cluster_id
                                   ORDER BY f.detection_confidence DESC
                               ) AS rn
                        FROM faces f
                        JOIN photos p ON p.id = f.photo_id
                        WHERE f.cluster_id IN ({placeholders})
                          AND f.dismissed = 0
                    ) WHERE rn <= 6""",
                cluster_ids,
            )
            faces_by_cluster: dict[int, list[dict]] = {c.id: [] for c in clusters}
            for r in cur:
                faces_by_cluster[r[0]].append({
                    "face_id": r[1], "photo_id": r[2], "photo_path": r[3],
                    "bbox_x": r[4], "bbox_y": r[5], "bbox_w": r[6], "bbox_h": r[7],
                    "confidence": r[8],
                })
            for c in clusters:
                cluster_data.append({"cluster": c, "faces": faces_by_cluster[c.id]})
        return render_template("review.html", cluster_data=cluster_data, active_page="review")

    @app.route("/timeline")
    def timeline():
        conn = get_conn()
        # Fetch photos with faces grouped by date
        cur = conn.execute(
            """SELECT p.exif_date, p.id, p.path, f.id AS face_id,
                      f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                      f.detection_confidence, f.cluster_id, c.name AS cluster_name
               FROM photos p
               JOIN faces f ON f.photo_id = p.id AND f.dismissed = 0
               LEFT JOIN clusters c ON c.id = f.cluster_id
               WHERE p.exif_date IS NOT NULL
               ORDER BY p.exif_date DESC, f.detection_confidence DESC"""
        )
        # Group by date (day)
        from collections import OrderedDict

        timeline_data = OrderedDict()
        for r in cur:
            date_str = r[0][:10] if r[0] else "Unknown"
            if date_str not in timeline_data:
                timeline_data[date_str] = []
            timeline_data[date_str].append({
                "photo_id": r[1], "photo_path": r[2], "face_id": r[3],
                "bbox_x": r[4], "bbox_y": r[5], "bbox_w": r[6], "bbox_h": r[7],
                "confidence": r[8], "cluster_id": r[9],
                "cluster_name": r[10] or "Unclustered",
            })
        return render_template(
            "timeline.html", timeline_data=timeline_data,
            active_page="timeline",
        )

    @app.route("/dismissed")
    def dismissed():
        conn = get_conn()
        faces = get_dismissed_faces(conn)
        return render_template("dismissed.html", faces=faces, active_page="dismissed")

    # ── Asset serving ──────────────────────────────────────────────────

    @app.route("/thumb/<int:face_id>")
    def thumbnail(face_id):
        conn = get_conn()
        face = get_face_by_id(conn, face_id)
        if face is None:
            abort(404)
        path = get_or_create_thumbnail(
            scan_root, face_id, face["photo_path"],
            face["bbox_x"], face["bbox_y"], face["bbox_w"], face["bbox_h"],
        )
        if path is None or not path.exists():
            abort(404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/photo/<int:face_id>")
    def photo(face_id):
        conn = get_conn()
        face = get_face_by_id(conn, face_id)
        if face is None:
            abort(404)
        photo_path = Path(face["photo_path"])
        if not photo_path.exists():
            abort(404)
        return send_file(photo_path)

    # ── Cluster management API ─────────────────────────────────────────

    @app.route("/api/rename", methods=["POST"])
    def api_rename():
        data = request.get_json(force=True)
        try:
            cid = _require_int(data.get("cluster_id"), "cluster_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        new_name = data.get("name", "").strip()
        if not new_name:
            return jsonify({"error": "name required"}), 400
        conn = get_conn()
        try:
            result = actions.rename_person_full(conn, cid, new_name)
        except actions.ActionError as e:
            log.warning("api_rename failed for cluster %d: %s", cid, e)
            return jsonify({"error": str(e)}), e.status
        if result is None:
            return jsonify({"error": "cluster not found"}), 404
        return jsonify({"ok": True, "cluster_id": cid, **result})

    @app.route("/api/merge", methods=["POST"])
    def api_merge():
        data = request.get_json(force=True)
        try:
            keep = _require_int(data.get("keep_id"), "keep_id")
            merge = _require_int(data.get("merge_id"), "merge_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        try:
            actions.merge_people(conn, keep, merge)
        except actions.ActionError as e:
            log.warning("api_merge failed (keep=%d, merge=%d): %s", keep, merge, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "kept": keep, "merged": merge})

    @app.route("/api/split", methods=["POST"])
    def api_split():
        data = request.get_json(force=True)
        try:
            fid = _require_int(data.get("face_id"), "face_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        new_name = data.get("name", "").strip()
        conn = get_conn()
        try:
            result = actions.split_face(conn, fid, new_name)
        except actions.ActionError as e:
            log.warning("api_split failed for face %d: %s", fid, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "face_id": fid, "new_cluster_id": result["cluster_id"]})

    @app.route("/api/split-batch", methods=["POST"])
    def api_split_batch():
        """Split multiple faces into similarity-grouped new clusters."""
        data = request.get_json(force=True)
        face_ids = data.get("face_ids")
        if not isinstance(face_ids, list) or not face_ids:
            return jsonify({"error": "face_ids must be a non-empty list"}), 400
        try:
            face_ids = [_require_int(fid, "face_id") for fid in face_ids]
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        base_name = data.get("name") or ""
        threshold = data.get("threshold")
        try:
            eps = float(threshold) if threshold is not None else DEFAULT_CLUSTER_THRESHOLD
        except (TypeError, ValueError):
            return jsonify({"error": "threshold must be a number"}), 400
        conn = get_conn()
        try:
            result = actions.split_faces_batch(conn, face_ids, base_name, eps=eps)
        except actions.ActionError as e:
            log.warning("api_split_batch failed for faces %s: %s", face_ids, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "new_cluster_ids": result["cluster_ids"]})

    @app.route("/api/move-face", methods=["POST"])
    def api_move_face():
        """Move a face to an existing cluster (for drag-and-drop)."""
        data = request.get_json(force=True)
        try:
            fid = _require_int(data.get("face_id"), "face_id")
            target_cluster = _require_int(data.get("target_cluster_id"), "target_cluster_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        face = get_face_by_id(conn, fid)
        if face is None:
            return jsonify({"error": "face not found"}), 404
        cluster = get_cluster_by_id(conn, target_cluster)
        if cluster is None:
            return jsonify({"error": "target cluster not found"}), 404
        update_face_cluster(conn, fid, target_cluster)
        return jsonify({
            "ok": True, "face_id": fid,
            "target_cluster_id": target_cluster, "target_name": cluster.name,
        })

    @app.route("/api/move-faces", methods=["POST"])
    def api_move_faces():
        """Move multiple faces to an existing cluster (batch drag-and-drop)."""
        data = request.get_json(force=True)
        face_ids = data.get("face_ids", [])
        if not face_ids or not isinstance(face_ids, list):
            return jsonify({"error": "face_ids list required"}), 400
        try:
            target_cluster = _require_int(data.get("target_cluster_id"), "target_cluster_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        cluster = get_cluster_by_id(conn, target_cluster)
        if cluster is None:
            return jsonify({"error": "target cluster not found"}), 404
        assignments = [(target_cluster, int(fid)) for fid in face_ids]
        update_face_clusters_batch(conn, assignments)
        return jsonify({
            "ok": True, "moved": len(face_ids),
            "target_cluster_id": target_cluster, "target_name": cluster.name,
        })

    @app.route("/api/recluster", methods=["POST"])
    def api_recluster():
        """Re-cluster a single cluster into sub-clusters (background task)."""
        data = request.get_json(force=True)
        try:
            cid = _require_int(data.get("cluster_id"), "cluster_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        threshold = float(data.get("threshold", DEFAULT_CLUSTER_THRESHOLD))

        task = create_task("recluster")

        def do_recluster():
            conn = sqlite3.connect(str(db_path))
            configure_connection(conn)
            try:
                result = actions.recluster_person(conn, cid, eps=threshold)
                task.result = result
            finally:
                conn.close()

        run_in_background(task, do_recluster)
        return jsonify({"ok": True, "task_id": task.id})

    @app.route("/api/dismiss-cluster", methods=["POST"])
    def api_dismiss_cluster():
        """Dismiss all faces in a cluster and delete it."""
        data = request.get_json(force=True)
        try:
            cid = _require_int(data.get("cluster_id"), "cluster_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        try:
            count = actions.dismiss_cluster(conn, cid)
        except actions.ActionError as e:
            log.warning("api_dismiss_cluster failed for cluster %d: %s", cid, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "cluster_id": cid, "faces_dismissed": count})

    @app.route("/api/dismiss", methods=["POST"])
    def api_dismiss():
        """Mark a face as not-a-face (false positive)."""
        data = request.get_json(force=True)
        try:
            fid = _require_int(data.get("face_id"), "face_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        try:
            actions.dismiss_face(conn, fid)
        except actions.ActionError as e:
            log.warning("api_dismiss failed for face %d: %s", fid, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "face_id": fid})

    @app.route("/api/restore", methods=["POST"])
    def api_restore():
        """Undo a dismiss — mark the face as active again."""
        data = request.get_json(force=True)
        try:
            fid = _require_int(data.get("face_id"), "face_id")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        conn = get_conn()
        try:
            actions.restore_face(conn, fid)
        except actions.ActionError as e:
            log.warning("api_restore failed for face %d: %s", fid, e)
            return jsonify({"error": str(e)}), e.status
        return jsonify({"ok": True, "face_id": fid})

    # ── Filesystem browser ─────────────────────────────────────────────

    @app.route("/api/browse")
    def api_browse():
        """Return directory listing for local filesystem navigation."""
        raw = request.args.get("path", "").strip()
        target = Path(raw).expanduser() if raw else scan_root
        try:
            target = target.resolve()
        except Exception:
            return jsonify({"error": "invalid path"}), 400
        if not target.is_dir():
            return jsonify({"error": "not a directory"}), 400
        try:
            entries = list(target.iterdir())
        except PermissionError:
            return jsonify({"error": "permission denied"}), 403
        dirs = sorted(
            [{"name": e.name, "path": str(e)} for e in entries if e.is_dir()],
            key=lambda d: d["name"].lower(),
        )
        parent = str(target.parent) if target != target.parent else None
        return jsonify({"current": str(target), "parent": parent, "dirs": dirs})

    # ── Long-running operations API ────────────────────────────────────

    @app.route("/api/scan", methods=["POST"])
    def api_scan():
        """Start a background scan."""
        from faceorganizer.scanner.scan_runner import ScanProgress, run_scan

        data = request.get_json(force=True) if request.is_json else {}
        workers_raw = data.get("workers")
        max_workers = None
        if workers_raw is not None:
            try:
                max_workers = max(1, min(16, int(workers_raw)))
            except (TypeError, ValueError):
                pass

        s = app.config["SETTINGS"]
        # Fall back to the settings default_workers if not supplied by caller
        if max_workers is None and s.default_workers is not None:
            max_workers = s.default_workers

        task = create_task("scan")
        progress = ScanProgress()
        task.progress = progress

        def do_scan():
            # Apply detection settings before worker threads are created.
            # Safe: single-user app, scan tasks are not concurrent.
            import faceorganizer.scanner.face_detector as _fd
            _fd.MIN_DETECTION_CONFIDENCE = s.detection_confidence
            _fd.MIN_FACE_SIZE = s.min_face_size

            conn = sqlite3.connect(str(db_path))
            configure_connection(conn)
            try:
                run_scan(conn, scan_root, parallel=True,
                         max_workers=max_workers, progress=progress,
                         stop_event=task.stop_event)
            finally:
                conn.close()

        run_in_background(task, do_scan)
        return jsonify({"ok": True, "task_id": task.id})

    @app.route("/api/cluster", methods=["POST"])
    def api_cluster():
        """Start background clustering.

        Pass ``{"incremental": true}`` to only cluster unassigned faces,
        preserving existing cluster names and merges.
        """
        data = request.get_json(force=True) if request.is_json else {}
        threshold = float(data.get("threshold", DEFAULT_CLUSTER_THRESHOLD))
        maybe_threshold = data.get("maybe_threshold")
        if maybe_threshold is not None:
            maybe_threshold = float(maybe_threshold)
        incremental = bool(data.get("incremental", False))

        task = create_task("cluster")

        def do_cluster():
            conn = sqlite3.connect(str(db_path))
            configure_connection(conn)
            try:
                if incremental:
                    from faceorganizer.clustering.cluster import run_incremental_clustering

                    result = run_incremental_clustering(
                        conn, eps=threshold, maybe_eps=maybe_threshold,
                    )
                    task.result = result
                else:
                    from faceorganizer.clustering.cluster import run_clustering

                    n = run_clustering(conn, eps=threshold)
                    task.result = {"clusters": n}
            finally:
                conn.close()

        run_in_background(task, do_cluster)
        return jsonify({"ok": True, "task_id": task.id})

    @app.route("/api/export", methods=["POST"])
    def api_export():
        """Start background export."""
        data = request.get_json(force=True)
        output_dir = data.get("output_dir", "").strip()
        if not output_dir:
            return jsonify({"error": "output_dir required"}), 400

        use_symlink = bool(data.get("symlink", False))
        task = create_task("export")

        def do_export():
            conn = sqlite3.connect(str(db_path))
            configure_connection(conn)
            try:
                summary = actions.export_people(
                    conn, Path(output_dir), symlink=use_symlink,
                    stop_event=task.stop_event,
                )
                task.result = {
                    "total": sum(summary.values()),
                    "people": len(summary),
                }
            finally:
                conn.close()

        run_in_background(task, do_export)
        return jsonify({"ok": True, "task_id": task.id})

    @app.route("/api/stats")
    def api_stats():
        """Get current database stats."""
        conn = get_conn()
        stats = get_scan_stats(conn)
        return jsonify(stats)

    @app.route("/api/task/<task_id>")
    def api_task_status(task_id):
        """Poll a background task's status."""
        task = get_task(task_id)
        if task is None:
            return jsonify({"error": "task not found"}), 404

        resp = {"id": task.id, "type": task.type, "status": task.status, "message": task.message}

        # Attach scan progress if available
        if hasattr(task, "progress") and task.progress is not None:
            p = task.progress
            if hasattr(p, "total"):
                resp["progress"] = {
                    "total": p.total,
                    "processed": p.processed,
                    "skipped": p.skipped,
                    "errors": p.errors,
                    "faces_found": p.faces_found,
                    "current_file": p.current_file,
                    "done": p.done,
                    "cancelled": getattr(p, "cancelled", False),
                }

        if task.result is not None:
            resp["result"] = task.result

        return jsonify(resp)

    @app.route("/api/task/<task_id>/stream")
    def api_task_stream(task_id):
        """SSE stream for real-time task progress updates."""

        def generate():
            while True:
                task = get_task(task_id)
                if task is None:
                    yield f"data: {json.dumps({'error': 'task not found'})}\n\n"
                    break

                resp = {
                    "id": task.id, "type": task.type,
                    "status": task.status, "message": task.message,
                }
                if hasattr(task, "progress") and task.progress is not None:
                    p = task.progress
                    if hasattr(p, "total"):
                        resp["progress"] = {
                            "total": p.total, "processed": p.processed,
                            "skipped": p.skipped, "errors": p.errors,
                            "faces_found": p.faces_found,
                            "current_file": p.current_file, "done": p.done,
                            "cancelled": getattr(p, "cancelled", False),
                        }
                if task.result is not None:
                    resp["result"] = task.result

                yield f"data: {json.dumps(resp)}\n\n"

                if task.status in ("done", "error", "cancelled"):
                    break
                time.sleep(1.0)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/task/<task_id>/cancel", methods=["POST"])
    def api_task_cancel(task_id):
        """Signal a running task to stop at the next checkpoint."""
        task = get_task(task_id)
        if task is None:
            return jsonify({"error": "task not found"}), 404
        task.stop_event.set()
        return jsonify({"ok": True})

    # ── Settings ───────────────────────────────────────────────────────

    @app.route("/settings")
    def settings_page():
        s = app.config["SETTINGS"]
        return render_template("settings.html", settings=s, active_page="settings")

    @app.route("/api/settings", methods=["GET"])
    def api_get_settings():
        s = app.config["SETTINGS"]
        return jsonify({
            "detection_confidence": s.detection_confidence,
            "min_face_size": s.min_face_size,
            "default_workers": s.default_workers,
        })

    @app.route("/api/settings", methods=["POST"])
    def api_post_settings():
        data = request.get_json(force=True)
        s = app.config["SETTINGS"]
        errors = []

        if "detection_confidence" in data:
            try:
                val = float(data["detection_confidence"])
                if not (0.1 <= val <= 1.0):
                    errors.append("detection_confidence must be 0.1–1.0")
                else:
                    s.detection_confidence = val
            except (TypeError, ValueError):
                errors.append("detection_confidence must be a number")

        if "min_face_size" in data:
            try:
                val = int(data["min_face_size"])
                if not (5 <= val <= 500):
                    errors.append("min_face_size must be 5–500")
                else:
                    s.min_face_size = val
            except (TypeError, ValueError):
                errors.append("min_face_size must be an integer")

        if "default_workers" in data:
            raw = data["default_workers"]
            try:
                s.default_workers = max(1, min(16, int(raw))) if raw is not None else None
            except (TypeError, ValueError):
                errors.append("default_workers must be an integer or null")

        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        s.save(data_dir)
        app.config["SETTINGS"] = s
        return jsonify({"ok": True})

    # ── Logs ───────────────────────────────────────────────────────────

    @app.route("/logs")
    def logs_page():
        return render_template("logs.html", active_page="logs")

    @app.route("/api/logs")
    def api_logs():
        log_path = data_dir / "faceorganizer.log"
        if not log_path.exists():
            return jsonify({"lines": [], "exists": False})
        try:
            n = min(2000, max(10, int(request.args.get("lines", 200))))
        except (ValueError, TypeError):
            n = 200
        try:
            with log_path.open(encoding="utf-8", errors="replace") as fh:
                all_lines = fh.readlines()
            lines = [line.rstrip("\n") for line in all_lines[-n:]]
            return jsonify({"lines": lines, "exists": True, "path": str(log_path)})
        except OSError as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/logs/clear", methods=["POST"])
    def api_logs_clear():
        log_path = data_dir / "faceorganizer.log"
        if not log_path.exists():
            return jsonify({"ok": True})
        try:
            log_path.write_text("", encoding="utf-8")
            return jsonify({"ok": True})
        except OSError as e:
            return jsonify({"error": str(e)}), 500

    return app
