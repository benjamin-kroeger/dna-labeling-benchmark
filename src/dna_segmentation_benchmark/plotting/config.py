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
    "boundary_shift_distribution": PlotMetadata(
        display_name="Boundary Shift Distribution",
        description="Distribution of splice-site boundary errors among BOUNDARY_SHIFT transcripts "
        "(same segment count, ≥1 position shifted).",
        bullet_points=(
            "Left: number of shifted boundary positions per transcript",
            "Middle: total absolute bp offset summed across shifted positions",
            "Right: scatter — count vs total bp offset per transcript",
        ),
        caveat="Only transcripts classified as BOUNDARY_SHIFT are included. "
        "Transcripts with no boundary shifts (exact, missed, etc.) are excluded.",
    ),
    "ts_level_precision": PlotMetadata(
        display_name="Transcript-Level Precision metrics",
        description="Precision across various metrics. All metrics use double penalties, so for intron chain precision = recall",
        bullet_points=(
            "Intron Chain: All predicted introns match the gt introns",
            "Exact transcript: All predicted exons match",
            "Superset: All predicted exons match, or are novel exons",
            "Subset: All predicted exons are part of the gt transcript",
        ),
    ),
    "ts_level_recall": PlotMetadata(
        display_name="Transcript-Level Recall metrics",
        description="Recall across various metrics. All metrics use double penalties, so for intron chain precision = recall",
        bullet_points=(
            "Intron Chain: All predicted introns match the gt introns",
            "Exact transcript: All predicted exons match",
            "Superset: All predicted exons match, or are novel exons",
            "Subset: All predicted exons are part of the gt transcript",
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
    "per_transcript_soft_exon": PlotMetadata(
        display_name="Per-transcript Soft Exon Metrics",
        description="Continuous per-transcript view of structural quality, "
        "complementing the strict all-or-nothing intron_chain and "
        "the corpus-averaged perfect_boundary_hit metrics.",
        bullet_points=(
            "Left: fraction of GT exons whose (start, end) is recovered exactly "
            "— a transcript with 9/10 exons right scores 0.9",
            "Right: count of predicted exons per transcript whose (start, end) "
            "does not match any GT exon (hallucinations)",
            "Histograms are overlayed across methods for direct comparison",
        ),
        caveat="Only transcripts with at least one GT exon are included. "
        "A near-zero recall mass with a fat right tail of hallucinations "
        "indicates a model that guesses without recovering true structure.",
    ),
    # --- Transcript-level P/R tiers ---
    "transcript_exact": PlotMetadata(
        display_name="Transcript Exact Match",
        description="Precision and recall at the transcript level requiring identical segment chains.",
        show_tp_tn_fp_fn=True,
        tp_definition="GT transcript exactly reproduced by prediction",
        fp_definition="Prediction exists but does not exactly match GT",
        fn_definition="GT transcript not exactly matched by any prediction",
    ),
    "pred_is_superset": PlotMetadata(
        display_name="Transcript All GT Found (Superset)",
        description="Recall-oriented: all GT segments are present in the prediction (pred may have extras).",
        show_tp_tn_fp_fn=True,
        tp_definition="All GT segments present in prediction (pred is superset or equal)",
        fp_definition="Prediction exists but misses at least one GT segment",
        fn_definition="GT transcript not fully recovered",
    ),
    "pred_is_subset": PlotMetadata(
        display_name="Transcript All Pred Valid (Subset)",
        description="Precision-oriented: all predicted segments correspond to GT segments (some GT segments may be missing).",
        show_tp_tn_fp_fn=True,
        tp_definition="All predicted segments are valid GT segments (pred is subset or equal)",
        fp_definition="Prediction contains segments not in GT",
        fn_definition="GT transcript not matched at this level",
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
        description="Boundary prediction error for position in coding span, if a predicted exon does not perfectly"
        "match the gt, all the bins with in the coding span of set exon are incremented by 1",
        caveat="Greedy matching for maximum overlap is used to match 2 exons ",
    ),
}
