"""Stage 1: Load raw arrays from a SolverAdapter."""


def harvest(adapter, split: str) -> dict:
    """Call adapter.load_split(split), return data dict."""
    data = adapter.load_split(split)
    return data
