"""CLI entry point and commands for FaceOrganizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faceorganizer import __version__
from faceorganizer.config import DEFAULT_CLUSTER_THRESHOLD
from faceorganizer.logging_config import get_logger, setup_logging

log = get_logger("cli")


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan a folder for photos, detect faces, and store results."""
    from tqdm import tqdm

    from faceorganizer.config import get_db_path
    from faceorganizer.database.core import get_scan_stats
    from faceorganizer.database.schema import init_db
    from faceorganizer.scanner.scan_runner import ScanProgress, run_scan

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        log.error("'%s' is not a directory", folder)
        sys.exit(1)

    db_path = get_db_path(folder)
    conn = init_db(db_path)
    log.info("Scan target: %s  Database: %s", folder, db_path)
    print(f"Discovering images in {folder} ...")

    progress = ScanProgress()
    pbar = [None]  # mutable container for closure

    def on_progress(p: ScanProgress):
        if pbar[0] is None and p.total > 0:
            pbar[0] = tqdm(total=p.total, desc="Detecting faces", unit="photo")
            if p.skipped > 0:
                pbar[0].update(p.skipped)
        if pbar[0] is not None:
            current = p.processed + p.skipped
            pbar[0].n = current
            pbar[0].refresh()

    use_parallel = not getattr(args, "no_parallel", False)
    run_scan(
        conn, folder,
        recursive=not args.no_recursive,
        parallel=use_parallel,
        max_workers=getattr(args, "workers", None),
        progress=progress,
        on_progress=on_progress,
    )

    if pbar[0] is not None:
        pbar[0].close()
    conn.close()

    print(f"\nScan complete: {progress.processed} processed, "
          f"{progress.skipped} skipped, {progress.errors} errors, "
          f"{progress.faces_found} faces detected")

    conn = init_db(db_path)
    s = get_scan_stats(conn)
    conn.close()
    print(f"Total in DB: {s['photos']} photos, {s['faces']} faces, "
          f"{s['clusters']} clusters")


def cmd_cluster(args: argparse.Namespace) -> None:
    """Run DBSCAN clustering on stored face embeddings."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.schema import init_db

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)

    if args.incremental:
        from faceorganizer.clustering.cluster import run_incremental_clustering

        result = run_incremental_clustering(conn, eps=args.threshold)
        conn.close()
        print(
            f"Incremental clustering: {result['assigned']} assigned to existing, "
            f"{result['new_clusters']} new clusters, "
            f"{result['new_noise']} still unassigned"
        )
    else:
        from faceorganizer.clustering.cluster import run_clustering

        num_clusters = run_clustering(conn, eps=args.threshold)
        conn.close()
        if num_clusters == 0:
            print("No clusters formed. Try lowering --threshold or scanning more photos.")
        else:
            print(f"Clustering complete: {num_clusters} people identified")


def cmd_show(args: argparse.Namespace) -> None:
    """Print a summary of clusters: N people, M faces, top people by count."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.core import get_clusters, get_scan_stats
    from faceorganizer.database.schema import init_db

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    stats = get_scan_stats(conn)
    clusters = get_clusters(conn)
    conn.close()

    print(f"Database: {db_path}")
    print(f"  Photos: {stats['photos']}  |  Faces: {stats['faces']}  |  "
          f"People: {stats['clusters']}  |  Unclustered: {stats['unclustered_faces']}")

    if not clusters:
        print("\nNo clusters yet. Run 'cluster' first.")
        return

    print("\nTop people by face count:")
    for c in clusters[:20]:
        print(f"  {c.name:<20s}  {c.face_count} face(s)")

    if len(clusters) > 20:
        print(f"  ... and {len(clusters) - 20} more")


def cmd_export(args: argparse.Namespace) -> None:
    """Export photos organized into per-person folders."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.schema import init_db
    from faceorganizer.organizer.export import export_by_person

    folder = Path(args.folder).resolve()
    output = Path(args.output).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    summary = export_by_person(conn, output, symlink=args.symlink)
    conn.close()

    if not summary:
        print("Nothing to export. Run 'cluster' first.")
        return

    total = sum(summary.values())
    print(f"Exported {total} photos into {len(summary)} person folders at {output}")


def _open_db_or_exit(folder: Path):
    """Resolve the folder's database, exiting with an error if it hasn't been scanned yet."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.schema import init_db

    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)
    return init_db(db_path)


def cmd_rename(args: argparse.Namespace) -> None:
    """Rename a cluster."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    ok = actions.rename_person(conn, args.cluster_id, args.name)
    conn.close()

    if ok:
        print(f"Cluster {args.cluster_id} renamed to '{args.name}'")
    else:
        print(f"Cluster {args.cluster_id} not found")
        sys.exit(1)


def cmd_dismiss(args: argparse.Namespace) -> None:
    """Dismiss a face as not-a-face (false positive detection)."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    try:
        if args.restore:
            actions.restore_face(conn, args.face_id)
            action = "restored"
        else:
            actions.dismiss_face(conn, args.face_id)
            action = "dismissed"
    except actions.ActionError as e:
        print(str(e))
        sys.exit(1)
    finally:
        conn.close()

    print(f"Face {args.face_id} {action}")


def cmd_dismiss_cluster(args: argparse.Namespace) -> None:
    """Dismiss every face in a cluster and delete it."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    try:
        count = actions.dismiss_cluster(conn, args.cluster_id)
    except actions.ActionError as e:
        print(str(e))
        sys.exit(1)
    finally:
        conn.close()

    print(f"Cluster {args.cluster_id} dismissed: {count} face(s) marked as not-a-face")


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge one cluster into another."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    try:
        actions.merge_people(conn, args.keep_id, args.merge_id)
    except actions.ActionError as e:
        print(str(e))
        sys.exit(1)
    finally:
        conn.close()

    print(f"Cluster {args.merge_id} merged into cluster {args.keep_id}")


def cmd_split(args: argparse.Namespace) -> None:
    """Split a single face out of its cluster into a new one."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    try:
        result = actions.split_face(conn, args.face_id, args.name or "")
    except actions.ActionError as e:
        print(str(e))
        sys.exit(1)
    finally:
        conn.close()

    print(
        f"Face {args.face_id} split into new cluster {result['cluster_id']} "
        f"('{result['name']}')"
    )


def cmd_recluster(args: argparse.Namespace) -> None:
    """Re-cluster a single cluster into sub-clusters."""
    from faceorganizer import actions

    folder = Path(args.folder).resolve()
    conn = _open_db_or_exit(folder)
    try:
        result = actions.recluster_person(conn, args.cluster_id, eps=args.threshold)
    except actions.ActionError as e:
        log.error(str(e))
        sys.exit(1)
    finally:
        conn.close()

    if result["new_clusters"] == 0 and result["noise"] == 0:
        print("Cluster is already cohesive — no changes made.")
    else:
        print(
            f"Recluster complete: {result['new_clusters']} new sub-clusters, "
            f"{result['noise']} faces became unassigned"
        )


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch the web review UI."""
    import threading
    import webbrowser

    from faceorganizer.config import get_data_dir

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        log.error("'%s' is not a directory", folder)
        sys.exit(1)

    setup_logging(args.verbose, log_dir=get_data_dir(folder))

    from faceorganizer.web.app import create_app

    app = create_app(folder)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting web UI at {url}")

    if not getattr(args, "no_browser", False):
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    # host is deliberately hardcoded to 127.0.0.1, not configurable via a
    # flag: the app has no authentication, and /api/browse exposes an
    # unrestricted local filesystem listing (see its docstring in web/app.py).
    # Both are fine for a single-user localhost tool but would become a real
    # exposure if this were ever bound to a network-visible address. --debug
    # additionally enables Werkzeug's interactive debugger/reloader, which is
    # also localhost-only-safe but should not be combined with wider binding.
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show database statistics."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.core import get_scan_stats
    from faceorganizer.database.schema import init_db

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    s = get_scan_stats(conn)
    conn.close()

    print(f"Database: {db_path}")
    print(f"  Photos:            {s['photos']}")
    print(f"  Faces:             {s['faces']}")
    print(f"  Clusters:          {s['clusters']}")
    print(f"  Unclustered faces: {s['unclustered_faces']}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="faceorganizer",
        description="Detect, cluster, and organize photos by face.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Scan a folder for faces")
    p_scan.add_argument("folder", help="Path to the folder of photos")
    p_scan.add_argument(
        "--no-recursive", action="store_true", help="Don't scan subdirectories"
    )
    p_scan.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel face detection (use single process)",
    )
    p_scan.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Number of parallel worker threads (default: min(4, cpu_count-1))",
    )
    p_scan.set_defaults(func=cmd_scan)

    # cluster
    p_cluster = sub.add_parser("cluster", help="Cluster faces into people")
    p_cluster.add_argument("folder", help="Path to the scanned folder")
    p_cluster.add_argument(
        "--threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD,
        help=f"DBSCAN cosine distance eps (default: {DEFAULT_CLUSTER_THRESHOLD}, lower = stricter)",
    )
    p_cluster.add_argument(
        "--incremental", action="store_true",
        help="Only cluster unassigned faces, preserving existing clusters and names",
    )
    p_cluster.set_defaults(func=cmd_cluster)

    # recluster
    p_recluster = sub.add_parser("recluster", help="Re-cluster a single cluster into sub-clusters")
    p_recluster.add_argument("folder", help="Path to the scanned folder")
    p_recluster.add_argument("cluster_id", type=int, help="Cluster ID to recluster")
    p_recluster.add_argument(
        "--threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD,
        help=f"DBSCAN cosine distance eps (default: {DEFAULT_CLUSTER_THRESHOLD}, lower = stricter)",
    )
    p_recluster.set_defaults(func=cmd_recluster)

    # merge
    p_merge = sub.add_parser("merge", help="Merge one cluster into another")
    p_merge.add_argument("folder", help="Path to the scanned folder")
    p_merge.add_argument("keep_id", type=int, help="Cluster ID to keep")
    p_merge.add_argument("merge_id", type=int, help="Cluster ID to merge into keep_id (deleted)")
    p_merge.set_defaults(func=cmd_merge)

    # split
    p_split = sub.add_parser("split", help="Split a single face into a new cluster")
    p_split.add_argument("folder", help="Path to the scanned folder")
    p_split.add_argument("face_id", type=int, help="Face ID to split out")
    p_split.add_argument(
        "name", nargs="?", default="",
        help="Name for the new cluster (default: Split_<face_id>)",
    )
    p_split.set_defaults(func=cmd_split)

    # show
    p_show = sub.add_parser("show", help="Show cluster summary")
    p_show.add_argument("folder", help="Path to the scanned folder")
    p_show.set_defaults(func=cmd_show)

    # export
    p_export = sub.add_parser("export", help="Export photos into per-person folders")
    p_export.add_argument("folder", help="Path to the scanned folder")
    p_export.add_argument("output", help="Output directory for organized photos")
    p_export.add_argument(
        "--symlink", action="store_true", help="Create symlinks instead of copying"
    )
    p_export.set_defaults(func=cmd_export)

    # rename
    p_rename = sub.add_parser("rename", help="Rename a person cluster")
    p_rename.add_argument("folder", help="Path to the scanned folder")
    p_rename.add_argument("cluster_id", type=int, help="Cluster ID to rename")
    p_rename.add_argument("name", help="New name for the cluster")
    p_rename.set_defaults(func=cmd_rename)

    # dismiss
    p_dismiss = sub.add_parser("dismiss", help="Mark a face as not-a-face (false positive)")
    p_dismiss.add_argument("folder", help="Path to the scanned folder")
    p_dismiss.add_argument("face_id", type=int, help="Face ID to dismiss")
    p_dismiss.add_argument(
        "--restore", action="store_true",
        help="Undo a dismiss — mark the face as active again",
    )
    p_dismiss.set_defaults(func=cmd_dismiss)

    # dismiss-cluster
    p_dismiss_cluster = sub.add_parser(
        "dismiss-cluster", help="Dismiss every face in a cluster and delete it"
    )
    p_dismiss_cluster.add_argument("folder", help="Path to the scanned folder")
    p_dismiss_cluster.add_argument("cluster_id", type=int, help="Cluster ID to dismiss")
    p_dismiss_cluster.set_defaults(func=cmd_dismiss_cluster)

    # serve
    p_serve = sub.add_parser("serve", help="Launch the web review UI")
    p_serve.add_argument("folder", help="Path to the scanned folder")
    p_serve.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    p_serve.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    p_serve.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    p_serve.set_defaults(func=cmd_serve)

    # stats
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("folder", help="Path to the scanned folder")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def _resolve_log_dir(args: argparse.Namespace) -> Path | None:
    """Determine the log file directory from the command's folder arg."""
    from faceorganizer.config import get_data_dir

    folder = getattr(args, "folder", None)
    if folder:
        return get_data_dir(Path(folder).resolve())
    return None


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose, log_dir=_resolve_log_dir(args))
    log.debug("FaceOrganizer %s starting (verbosity=%d)", __version__, args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
