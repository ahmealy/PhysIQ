"""Stage 5: Post-ingest hooks (e.g. rebuild confidence index)."""


def rebuild_confidence_index(result_dir: str = "result") -> bool:
    """
    Attempt to rebuild confidence index by calling existing build_confidence_index.
    Returns True if successful, False if skipped (no results or import error).
    """
    try:
        from result.confidence_index import build_confidence_index
        build_confidence_index(result_dir)
        return True
    except Exception:
        return False
