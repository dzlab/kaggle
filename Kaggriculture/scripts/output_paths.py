"""Compatibility exports for the shared output-path safety helpers."""

from kagriculture_agent.output_paths import (
    CANONICAL_PRODUCTION_FILE_NAMES,
    PROTECTED_OUTPUT_NAMES,
    PRODUCTION_DIRECTORY_NAMES,
    resolve_output_path,
    validate_training_output_path,
)

__all__ = [
    "CANONICAL_PRODUCTION_FILE_NAMES",
    "PROTECTED_OUTPUT_NAMES",
    "PRODUCTION_DIRECTORY_NAMES",
    "resolve_output_path",
    "validate_training_output_path",
]
