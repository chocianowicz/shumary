"""Shared exception types for the pipeline.

Library code should raise `ShumaryError` (or a subclass) instead of calling
`sys.exit` directly, so callers — especially `process_directory`, which
processes many files in a batch — can catch failures per-file and continue.
"""


class ShumaryError(Exception):
    """Raised for expected, user-facing pipeline failures."""
