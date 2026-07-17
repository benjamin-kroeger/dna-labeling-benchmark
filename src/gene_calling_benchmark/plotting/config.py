import dataclasses
from importlib.resources.abc import Traversable
from pathlib import Path

from importlib import resources

PACKAGE_NAME = "gene_calling_benchmark"
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
    secondary_icon_path : Path | Traversable | None
        Optional second PNG icon.  When set, it is rendered beside
        ``icon_path`` (for plots that pair two pictograms in one panel).
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
    tp_definition, tn_definition, fp_definition, fn_definition : str | None
        Per-metric text for the TP / TN / FP / FN block (used when
        ``show_tp_tn_fp_fn`` is set).
    """

    icon_path: Path | Traversable | None = None
    secondary_icon_path: Path | Traversable | None = None
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
    "indel_rates_by_boundary": PlotMetadata(
        display_name="INDEL Rate by GT Boundary",
        description="Per-method exon-position × event-type heatmap of error rate "
        "(events ÷ opportunities): the cross-method comparable INDEL view. "
        "Anchored slips, splits and whole_deletions divide by the count of GT "
        "exons of that position type; joined by GT intron count. whole_insertions "
        "are excluded — a detached false positive has no bounded GT opportunity.",
        caveat="Cells with no opportunity or zero events are masked grey. "
        "Rates normalise out corpus size, so methods on different inputs stay comparable.",
    ),
    "indel_counts_by_boundary": PlotMetadata(
        display_name="INDEL Counts by GT Boundary",
        description="Raw-magnitude companion to the rate heatmap: per-method GT "
        "boundary × event-type counts on a log colour scale, so one dominant cell "
        "(e.g. thousands of whole insertions) does not wash out the tens-count "
        "boundary slips.",
        caveat="Zero cells are masked. Absolute counts are not comparable across "
        "methods evaluated on different inputs — use the rate view for that.",
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
        description="Is the predicted section contained within its matched GT section "
        "(pred ⊆ GT, inclusive of an exact match)? Uses 1:1 greedy matching; the FP is "
        "hardened so a matched pair that over-extends past the GT is booked as both FP and FN.",
        show_tp_tn_fp_fn=True,
        tp_definition="Matched prediction lies within its GT section (pred ⊆ GT)",
        tn_definition="N/A",
        fp_definition="Prediction not contained in its matched GT (over-extends, or unmatched)",
        fn_definition="GT section whose matched prediction is not contained (or unmatched)",
    ),
    "full_coverage_hit": PlotMetadata(
        display_name="Full Coverage Hit Metrics",
        icon_path=ICON_PATH / "full_coverage.png",
        description="Does the predicted section fully span its matched GT section "
        "(pred ⊇ GT, inclusive of an exact match)? Uses 1:1 greedy matching; the FP is "
        "hardened so a matched pair that falls short of the GT is booked as both FP and FN.",
        show_tp_tn_fp_fn=True,
        tp_definition="Matched prediction fully spans its GT section (pred ⊇ GT)",
        tn_definition="N/A",
        fp_definition="Prediction does not cover its matched GT (falls short, or unmatched)",
        fn_definition="GT section whose matched prediction does not cover it (or unmatched)",
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
        description="Mean Intersection-over-Union across the greedy 1:1-matched (GT, prediction) section pairs.",
    ),
    "iou_distribution": PlotMetadata(
        display_name="IoU Distribution",
        icon_path=ICON_PATH / "iou.png",
        description="Distribution of per-section IoU scores across the matched pairs.",
    ),
    # Exon-length distribution distance
    "length_emd": PlotMetadata(
        display_name="Exon-Length Distribution Distance (EMD)",
        description="Per transcript, the 1-D Wasserstein (Earth Mover's) distance "
        "between the GT and predicted exon-length multisets, averaged across "
        "transcripts (bar = mean, error bar = SEM). Lower means the predicted "
        "exon-length profile is closer to the ground truth.",
        bullet_points=(
            "Unordered & unmatched: compares the distribution of exon lengths, "
            "blind to exon identity and genomic position.",
            "Catches length redistribution, fragmentation, and systematic size "
            "bias even when exon count and intron chain tie.",
            "Error bar = SEM over the per-transcript EMDs; separation between "
            "methods indicates a real length-profile difference.",
        ),
        caveat="Not a per-exon boundary metric — a single exon extension appears "
        "only as a faint shift here. Use the INDEL run-length / boundary-offset "
        "plots for directional per-boundary edits.",
    ),
    "boundary_bias_landscape": PlotMetadata(
        display_name="Boundary Bias",
        description="Signed 5'/3' boundary residual heatmap (Pred − GT), one subplot per method. "
        "Mass off the (0,0) crosshair reveals systematic over- and under-shifts at each exon edge.",
        bullet_points=(
            "Rows = 5' edge residual, columns = 3' edge residual (bp)",
            "Direction tells extension vs deletion per edge; the crosshair marks perfect matches",
            "Shared raw-count log color scale across methods — intensities are directly comparable",
        ),
    ),
    "boundary_recall_landscape": PlotMetadata(
        display_name="Cumulative Recall",
        description="Fraction of ground-truth boundaries recovered as the allowed 5' and 3' tolerance "
        "grows, one subplot per method. Tolerance is two-sided (±): a boundary counts as matched when "
        "its residual falls within ±k bp on that end — over- and under-shifts both count. Higher and "
        "faster-saturating is better.",
        bullet_points=(
            "Axes = 5'/3' tolerance budget in ±bp (total, both directions); each cell = recall within that budget",
            "Diagonal = symmetric tolerance relaxed equally on both ends",
            "Shared 0–1 color scale across methods for direct comparison",
        ),
    ),
    # Coding-phase drift
    "phase_drift": PlotMetadata(
        display_name="Coding-Phase Drift",
        description="Relative coding-phase drift (mod 3) between GT and predicted CDS at each "
        "co-CDS position. 0 means lockstep by CDS-base count; 1 or 2 means one annotation is "
        "ahead by that many bases. A structural comparison signal, not an absolute reading frame.",
        bullet_points=(
            "In-phase (0): pred and GT CDS-base counts agree mod 3 at this position",
            "Offset +1 / +2: one annotation is 1 or 2 bases ahead of the other",
            "Boundary indels in-frame: boundary indels whose length ≡ 0 (mod 3) — frame-preserving",
        ),
        caveat="Requires complete, in-frame CDS-only masks. Sequences with GT CDS length not "
        "divisible by 3 are excluded (n_skipped_non_divisible). UTR_CDS_INTRON mode only.",
    ),
    # --- Structural Coherence ---
    "boundary_shift_distribution": PlotMetadata(
        display_name="Boundary Shift Distribution",
        description="Per-boundary offset distributions for transcripts with correct chain "
        "topology (same segment count) but ≥1 shifted boundary. Given the exon count is right, "
        "how precisely is each junction placed? Complementary to the global boundary-precision "
        "landscape.",
        bullet_points=(
            "Left: ECDF of |offset| per method — fraction of misplaced junctions within k bp",
            "Middle: signed offset density — directional (5'/3') bias and the ±1/±2 bp spike",
            "Right: |offset| split by internal splice junction vs terminal TSS/TES",
        ),
        caveat="Only shifted boundaries (offset ≠ 0) from equal-count transcripts whose paired "
        "segments all overlap — conditional shift magnitude, not recall. Offsets are array-oriented "
        "(array-3' positive), not strand-resolved.",
    ),
    "ts_level_match_rate": PlotMetadata(
        display_name="Transcript-Level Match Rate (Multi-Exon Only)",
        description="Per-tier rate at which a predicted transcript's full chain (its set "
        "of intron/exon boundaries) matches GT. Each tier is all-or-nothing — the fraction "
        "of transcripts whose ENTIRE chain satisfies it. A chain mismatch is booked as both FP "
        "and FN, so precision = recall = F1; the precision-vs-recall contrast is carried by the "
        "subset vs superset tiers instead.",
        bullet_points=(
            "Exact intron-chain rate: entire intron boundary chain matches GT exactly",
            "Intron Subset: all predicted introns are real (pred ⊆ GT) — precision-flavoured",
            "Intron Superset: all GT introns recovered (pred ⊇ GT) — recall-flavoured",
            "Exact exon-chain rate: entire exon boundary chain matches GT exactly",
            "Exon Subset / Superset: analogous set semantics for exons",
            "Exon-chain tiers count MULTI-exon transcripts only — every bar covers "
            "transcripts with introns, apples-to-apples with the intron tiers. "
            "Single-exon genes are reported in their own plot.",
            "An intron is the gap between consecutive in-scope segments (gffcompare "
            "semantics). Under CDS scope these are CDS introns only — UTR introns are "
            "excluded, so both chains share the same scoped boundary set.",
        ),
    ),
    "single_exon_match_rate": PlotMetadata(
        display_name="Single-Exon Gene Match Rate",
        description="Fraction of single-exon genes whose single coding segment exactly matches GT "
        "(start and end both correct). Single-exon transcripts have no introns, so they are "
        "excluded from the multi-exon chain tiers; this plot reports them on their own. It is a "
        "sub-category of boundary exactness, surfaced separately so it does not dilute the "
        "multi-exon chain rates.",
        bullet_points=(
            "Single-exon exact match: the one coding segment's (start, end) equals GT exactly",
            "All-or-nothing per gene, so precision = recall = F1 (the representative match rate)",
        ),
    ),
    "transcript_match": PlotMetadata(
        display_name="Transcript Match Classification",
        description="Structural classification of each (GT, prediction) pair into 9 categories, "
        "ordered best→worst (the same order drives the green→red stacked bar).",
        bullet_points=(
            "exact: identical exon sets",
            "boundary_shift_internal: same exon count, all overlap; internal splice boundaries "
            "differ but locus span matches",
            "boundary_shift_terminal: same count, full overlap, but terminal (TSS/TES) boundaries differ",
            "missing_segments: pred ⊂ GT (all pred exons real, some GT exons absent)",
            "extra_segments: GT ⊂ pred (all GT exons found, pred has novel extras)",
            "partial_overlap: ≥1 exon (start,end) shared exactly, not a clean subset/superset",
            "substitution: no exon shared exactly, but exons overlap in bases (relocated)",
            "no_overlap: no shared exon and no base overlap at all",
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
    "per_transcript_exon_recovery": PlotMetadata(
        display_name="Per-transcript Exon Recovery",
        description="Continuous per-transcript view of structural quality, "
        "complementing the strict all-or-nothing exon_chain tiers and "
        "the corpus-averaged perfect_boundary_hit metrics.",
        bullet_points=(
            "Left: exon recall — fraction of GT exons whose (start, end) is "
            "recovered exactly; a transcript with 9/10 exons right scores 0.9",
            "Middle: exon precision — fraction of predicted exons whose "
            "(start, end) is an exact GT match",
            "Right: count of predicted exons per transcript whose (start, end) "
            "does not match any GT exon (false exons)",
            "Histograms are overlayed across methods for direct comparison",
        ),
        caveat="Only transcripts with at least one GT exon are included. "
        "A near-zero recall mass with a fat right tail of false exons "
        "indicates a model that guesses without recovering true structure.",
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
