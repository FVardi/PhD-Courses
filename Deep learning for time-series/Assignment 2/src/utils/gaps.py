# Gap detection has moved to src/evaluation/gaps.py.
# This shim re-exports everything so existing imports keep working.
from src.evaluation.gaps import (  # noqa: F401
    find_gaps,
    gaps_per_household,
    summarise_gaps,
    find_gap_lengths,
    gap_length_dataframe,
)
