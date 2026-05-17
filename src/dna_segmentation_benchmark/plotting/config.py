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
    "indel_counts": PlotMetadata(
        display_name="INDEL Counts",
        description="Stacked total of structural mismatches per method, "
        "broken down by INDEL category. Bars are sorted by total error count.",
        bullet_points=(
            "5'/3' extensions: prediction extends past one GT boundary",
            "5'/3' deletions: prediction stops short of one GT boundary",
            "joined: predicted span fuses two adjacent GT coding sections",
            "split: GT coding section fragmented into multiple predictions",
            "whole_insertions / whole_deletions: predicted/missed sections "
            "with no overlap to a GT counterpart",
        ),
        caveat="Counts are absolute mismatch-group counts across the corpus. "
        "Methods evaluated on different inputs are not directly comparable.",
    ),
    "indel_lengths": PlotMetadata(
        display_name="INDEL Length Distribution",
        description="Per-category length distribution of INDEL events on a "
        "log-scaled x-axis. One subplot per INDEL category; one histogram "
        "per method overlaid.",
        bullet_points=(
            "Long tails on extensions/deletions: occasional but very large boundary slips",
            "Heavy mass on whole_insertions/deletions: hallucinated or missed exons",
            "Tick labels are linear bp values even though bins are log-spaced",
        ),
        caveat="Only positive lengths are plotted (log scale). Zero-length "
        "events would not appear and indicate a different failure mode.",
    ),
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
    "boundary_landscape": PlotMetadata(
        display_name="Boundary Bias and cumulative Recall",
        description="Joint view of systematic boundary offsets and how recall recovers as 5' and 3' tolerance increases.",
        bullet_points=(
            "Left: signed 5'/3' residual heatmap showing systematic over- and under-shifts",
            "Right: cumulative recall surface across increasing total tolerance budgets on both boundaries",
        ),
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
        description="Distribution of splice-site boundary errors among transcripts where "
        "GT and pred have the same segment count but ≥1 boundary position differs.",
        bullet_points=(
            "Left: number of shifted boundary positions per transcript",
            "Middle: total absolute bp offset summed across shifted positions",
            "Right: scatter — count vs total bp offset per transcript",
        ),
        caveat="Only transcripts with at least one shifted boundary (count > 0) are included.",
    ),
    "ts_level_precision": PlotMetadata(
        display_name="Transcript-Level Precision metrics",
        description="Precision across transcript-level structural match tiers.",
        bullet_points=(
            "Exact intron chain: all intron boundaries match exactly",
            "Intron Subset: all predicted introns are real (pred ⊆ GT)",
            "Intron Superset: all GT introns recovered (pred ⊇ GT)",
            "Exact exon chain: all exon boundaries match exactly",
            "Exon Subset / Superset: analogous set semantics for exons",
            "Intron boundaries: DONOR.start → ACCEPTOR.end when splice-site "
            "labels are configured, matching gffcompare splice-junction semantics. "
            "Without splice-site labels, raw intron-segment boundaries are used.",
        ),
    ),
    "ts_level_recall": PlotMetadata(
        display_name="Transcript-Level Recall metrics",
        description="Recall across transcript-level structural match tiers.",
        bullet_points=(
            "Exact intron chain: all intron boundaries match exactly",
            "Intron Subset: all predicted introns are real (pred ⊆ GT)",
            "Intron Superset: all GT introns recovered (pred ⊇ GT)",
            "Exact exon chain: all exon boundaries match exactly",
            "Exon Subset / Superset: analogous set semantics for exons",
            "Intron boundaries: DONOR.start → ACCEPTOR.end when splice-site "
            "labels are configured, matching gffcompare splice-junction semantics. "
            "Without splice-site labels, raw intron-segment boundaries are used.",
        ),
    ),
    "transcript_match": PlotMetadata(
        display_name="Transcript Match Classification",
        description="Holistic structural classification of each (GT, prediction) pair into 8 granular categories.",
        bullet_points=(
            "exact: identical exon sets",
            "boundary_shift_internal: same count, splice-site boundaries differ but gene locus span matches",
            "boundary_shift_terminal: same count, terminal transcript boundaries also differ",
            "missing_segments: pred ⊂ GT (all pred exons are real, some GT exons absent)",
            "extra_segments: GT ⊂ pred (all GT exons found, pred has novel extras)",
            "partial_overlap: some exon (start,end) pairs shared, not a clean subset/superset",
            "no_overlap: no (start,end) pairs in common",
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
    # --- Exon chain P/R tiers ---
    "exon_chain": PlotMetadata(
        display_name="Exon Chain Exact Match",
        description="Precision and recall requiring identical exon segment boundary sets.",
        show_tp_tn_fp_fn=True,
        tp_definition="GT exon set exactly reproduced by prediction",
        fp_definition="Prediction exists but exon set does not exactly match GT",
        fn_definition="GT exon set not exactly matched",
    ),
    "exon_chain_superset": PlotMetadata(
        display_name="Exon Chain Superset",
        description="Recall-oriented: all GT exons are present in the prediction (pred may have extras).",
        show_tp_tn_fp_fn=True,
        tp_definition="All GT exons present in prediction (pred ⊇ GT)",
        fp_definition="Prediction exists but misses at least one GT exon",
        fn_definition="GT exon set not fully recovered",
    ),
    "exon_chain_subset": PlotMetadata(
        display_name="Exon Chain Subset",
        description="Precision-oriented: all predicted exons correspond to GT exons (some GT exons may be missing).",
        show_tp_tn_fp_fn=True,
        tp_definition="All predicted exons are valid GT exons (pred ⊆ GT)",
        fp_definition="Prediction contains exons not in GT",
        fn_definition="GT exon set not matched at this level",
    ),
    # --- Diagnostic Depth ---
    "position_bias": PlotMetadata(
        display_name="Position Bias",
        description="Per-position nucleotide prediction errors over the coding span, "
        "split into false negatives (GT coding missed by the prediction) and "
        "false positives (predicted coding inside the GT span absent from GT).",
        bullet_points=(
            "Left panel: false negatives — under-prediction density per position",
            "Right panel: false positives — over-prediction density per position",
            "Histograms are summed across sequences (longer sequences contribute more mass)",
        ),
        caveat="Predicted bases outside the GT coding span are clipped before binning, "
        "so the FP panel only reflects errors inside the gene locus.",
    ),
}
