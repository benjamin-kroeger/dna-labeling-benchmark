import dataclasses
from importlib.resources.abc import Traversable
from pathlib import Path

from importlib import resources

PACKAGE_NAME = "dna_segmentation_benchmark"
ICON_PATH = resources.files(PACKAGE_NAME) / "icons"
ICON_MAP = {
    "5_prime_extensions": ICON_PATH / "left_extension.png",
    "3_prime_extensions": ICON_PATH / "right_extension.png",
    "whole_insertions": ICON_PATH / "exon_insertion.png",
    "joined": ICON_PATH / "joined_exons.png",
    "5_prime_deletions": ICON_PATH / "left_deletion.png",
    "3_prime_deletions": ICON_PATH / "right_deletion.png",
    "whole_deletions": ICON_PATH / "exon_deletion.png",
    "split": ICON_PATH / "split_exons.png",
}

DEFAULT_FIG_SIZE = (16, 10)
DEFAULT_MULTI_PLOT_FIG_SIZE = (18, 12)


# ---------------------------------------------------------------------------
# Plot metadata — pictogram panel content
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlotMetadata:
    """Icon and explanatory text shown in the right-side pictogram panel.

    Attributes
    ----------
    icon_path : Path | Traversable | None
        Path to a PNG icon.  ``None`` means no icon yet.
    description : str
        Short paragraph explaining what the plot shows.
    bullet_points : tuple[str, ...] | None
        Optional bullet-point list rendered below the description.
        Each string is one bullet item (e.g. metric name + explanation).
    caveat : str | None
        Optional caveat or limitation.  Rendered in a distinct warning
        box below the description / bullets.
    display_name : str
        Human-readable title rendered above the icon.
    show_tp_tn_fp_fn : bool
        If ``True`` a compact TP / TN / FP / FN definitions block is
        rendered at the bottom of the panel.
    """

    icon_path: Path | Traversable | None = None
    description: str = ""
    bullet_points: tuple[str, ...] | None = None
    caveat: str | None = None
    display_name: str = ""
    show_tp_tn_fp_fn: bool = False
    tp_definition: str | None = None
    tn_definition: str | None = None
    fp_definition: str | None = None
    fn_definition: str | None = None


# Placeholder entries — fill in ``icon_path`` and ``description`` as
# pictograms are created.  Keys must match those used in
# :func:`compare_multiple_predictions`.
PLOT_METADATA: dict[str, PlotMetadata] = {
    # INDEL summary
    "indel_counts": PlotMetadata(display_name="INDEL Counts"),
    "indel_lengths": PlotMetadata(display_name="INDEL Length Distribution"),
    # ML precision / recall (one entry per level)
    "nucleotide": PlotMetadata(
        display_name="Nucleotide-Level Metrics",
        description="Per-base classification accuracy for the target class.",
        show_tp_tn_fp_fn=True,
        tp_definition="Nucleotides correctly predicted as the target class",
        tn_definition="Nucleotides correctly predicted as NOT the target class",
        fp_definition="Nucleotides incorrectly predicted as the target class",
        fn_definition="Nucleotides incorrectly predicted as NOT the target class",
    ),
    "neighborhood_hit": PlotMetadata(
        display_name="Neighborhood Hit Metrics",
        icon_path=ICON_PATH / "overlap.png",
        description="Do predicted sections overlap ground truth sections at all? "
                    "Uses 1:1 greedy matching by overlap length.",
        show_tp_tn_fp_fn=True,
        tp_definition="GT section matched to a prediction (any overlap)",
        tn_definition="N/A",
        fp_definition="Predicted section not matched to any GT section",
        fn_definition="GT section not matched to any prediction",
    ),
    "internal_hit": PlotMetadata(
        display_name="Internal Hit Metrics",
        icon_path=ICON_PATH / "internal.png",
        description="Is the matched prediction contained within the GT boundaries? "
                    "Forgiving to under-prediction. Uses 1:1 matching.",
        show_tp_tn_fp_fn=True,
        tp_definition="Matched prediction is completely contained within its GT section",
        tn_definition="N/A",
        fp_definition="Predicted section not matched to any GT section",
        fn_definition="GT section's matched prediction exceeds its boundaries (or is unmatched)",
    ),
    "full_coverage_hit": PlotMetadata(
        display_name="Full Coverage Hit Metrics",
        icon_path=ICON_PATH / "full_coverage.png",
        description="Does the matched prediction fully cover the GT section? "
                    "Forgiving to over-prediction. Uses 1:1 matching.",
        show_tp_tn_fp_fn=True,
        tp_definition="Matched prediction fully covers its GT section",
        tn_definition="N/A",
        fp_definition="Predicted section not matched to any GT section",
        fn_definition="GT section is not fully covered by its matched prediction (or is unmatched)",
    ),
    "perfect_boundary_hit": PlotMetadata(
        display_name="Perfect Boundary Hit Metrics",
        icon_path=ICON_PATH / "prefect_hit.png",
        description="Does the prediction exactly reproduce the GT boundaries (100% IoU)? "
                    "Uses sweep-based matching (no 1:1 constraint).",
        show_tp_tn_fp_fn=True,
        tp_definition="Matched prediction exactly matches its GT section's boundaries",
        tn_definition="N/A",
        fp_definition="Predicted section not matched to any GT section",
        fn_definition="GT section's matched prediction has inexact boundaries (or is unmatched)",
    ),
    # IoU
    "iou_average": PlotMetadata(
        display_name="Average IoU",
        icon_path=ICON_PATH / "iou.png",
        description="Mean Intersection-over-Union across all overlapping (GT, prediction) section pairs.",
    ),
    "iou_distribution": PlotMetadata(
        display_name="IoU Distribution",
        icon_path=ICON_PATH / "iou.png",
        description="Distribution of per-section IoU scores across all overlapping pairs.",
    ),
    # Frameshift
    "frameshift": PlotMetadata(
        display_name="Frameshift Distribution",
        description="Reading frame deviation (mod-3) between GT and predicted coding exons.",
        caveat="Only valid on single-transcript sequences with a coding label configured.",
    ),

    # --- Structural Coherence ---
    "gap_chain": PlotMetadata(
        display_name="Gap Chain Metrics",
        description="Compares the ordered gaps between consecutive segments. "
                    "For exons, gaps = introns (label-agnostic intron chain).",
        bullet_points=(
            "match_rate: fraction of sequences with identical gap chains",
            "count_match_rate: fraction with the same number of gaps",
            "lcs_ratio: ordering-aware partial credit via longest common subsequence",
        ),
    ),
    "transcript_match": PlotMetadata(
        display_name="Transcript Match Classification",
        description="Holistic structural classification of each (GT, prediction) pair.",
        bullet_points=(
            "exact: identical segment chains",
            "boundary_shift: same segment count, shifted boundaries",
            "missing_segments: pred is ordered subset of GT (segments skipped)",
            "extra_segments: GT is ordered subset of pred (segments inserted)",
            "structurally_different: very low similarity",
            "missed: no prediction for this class",
        ),
    ),
    "segment_count_delta": PlotMetadata(
        display_name="Segment Count Delta",
        description="Mean difference in segment counts (pred - GT) per method.",
        bullet_points=(
            "Red bars: over-segmentation (positive delta)",
            "Blue bars: under-segmentation (negative delta)",
            "Error bars: standard deviation across sequences",
        ),
    ),

    # --- Diagnostic Depth ---
    "junction_errors": PlotMetadata(
        display_name="Junction Error Taxonomy",
        description="Classifies each structural mismatch into a causal error type.",
        bullet_points=(
            "Exon skip: GT segments merged (intervening segment absent)",
            "Segment retention: GT segment absorbed by neighbours",
            "Novel insertion: extra segment splits a GT segment",
            "Cascade shift: boundary error propagates across 3+ segments",
            "Compensating errors: paired errors that cancel out",
        ),
    ),
    "position_bias": PlotMetadata(
        display_name="Position Bias",
        description="Match rate stratified by genomic position.",
        bullet_points=(
            "5' zone: first 25% of the sequence",
            "Interior: middle 50%",
            "3' zone: last 25%",
        ),
        caveat="A GT segment counts as 'matched' if any prediction overlaps it by >= 50%. "
               "This threshold is fixed and not configurable.",
    ),
}
