"""Golden-set evaluation helpers."""

from crosscheck.eval.golden_loader import discover_golden_files, load_golden_claims
from crosscheck.eval.metrics import ClaimScoreRow, print_summary, summarize, write_metrics_run_dir
from crosscheck.eval.retrieval_truth import score_retrieval_hit
from crosscheck.eval.sampler import sample_all_periods, sample_mini_golden

__all__ = [
    "ClaimScoreRow",
    "discover_golden_files",
    "extract_reference_figures",
    "load_golden_claims",
    "print_summary",
    "reference_overlap_hit",
    "sample_all_periods",
    "sample_mini_golden",
    "score_retrieval_hit",
    "summarize",
    "write_metrics_run_dir",
]
