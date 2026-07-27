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


def cmd_rename(args: argparse.Namespace) -> None:
    """Rename a cluster."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.schema import init_db
    from faceorganizer.organizer.naming import rename_person

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    ok = rename_person(conn, args.cluster_id, args.name)
    conn.close()

    if ok:
        print(f"Cluster {args.cluster_id} renamed to '{args.name}'")
    else:
        print(f"Cluster {args.cluster_id} not found")
        sys.exit(1)


def cmd_dismiss(args: argparse.Namespace) -> None:
    """Dismiss a face as not-a-face (false positive detection)."""
    from faceorganizer.config import get_db_path
    from faceorganizer.database.core import dismiss_face, restore_face
    from faceorganizer.database.schema import init_db

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    if args.restore:
        ok = restore_face(conn, args.face_id)
        action = "restored"
    else:
        ok = dismiss_face(conn, args.face_id)
        action = "dismissed"
    conn.close()

    if ok:
        print(f"Face {args.face_id} {action}")
    else:
        print(f"Face {args.face_id} not found")
        sys.exit(1)


def cmd_recluster(args: argparse.Namespace) -> None:
    """Re-cluster a single cluster into sub-clusters."""
    from faceorganizer.clustering.cluster import run_recluster
    from faceorganizer.config import get_db_path
    from faceorganizer.database.schema import init_db

    folder = Path(args.folder).resolve()
    db_path = get_db_path(folder)
    if not db_path.exists():
        log.error("No database found at %s. Run 'scan' first.", db_path)
        sys.exit(1)

    conn = init_db(db_path)
    try:
        result = run_recluster(conn, args.cluster_id, eps=args.threshold)
    except ValueError as e:
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
