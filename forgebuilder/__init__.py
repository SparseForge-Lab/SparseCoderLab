"""Dataset-v2 and repository-task infrastructure for SparseCoderLab.

This package defines offline schemas and deterministic storage/pipeline
primitives. It does not launch teacher models or execute untrusted repository
code by itself.
"""

from .schema import SCHEMA_VERSION, SchemaError, stable_sample_id, validate_record

__all__ = ["SCHEMA_VERSION", "SchemaError", "stable_sample_id", "validate_record"]
