import dataclasses
from importlib.abc import Traversable
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
        Short paragraph explaining what the plot shows.  Rendered as
        word-wrapped text below the icon.
    display_name : str
        Human-readable title rendered above the icon.
    show_tp_tn_fp_fn : bool
        If ``True`` a compact TP / TN / FP / FN definitions block is
        rendered at the bottom of the panel.
    """

    icon_path: Path | Traversable | None = None
    description: str = ""
    display_name: str = ""
    show_tp_tn_fp_fn: bool = False


# Placeholder entries — fill in ``icon_path`` and ``description`` as
# pictograms are created.  Keys must match those used in
# :func:`compare_multiple_predictions`.
PLOT_METADATA: dict[str, PlotMetadata] = {
    # INDEL summary
    "indel_counts": PlotMetadata(display_name="INDEL Counts"),
    "indel_lengths": PlotMetadata(display_name="INDEL Length Distribution"),
    # ML precision / recall (one entry per level)
    "ml_nucleotide_level_metrics": PlotMetadata(
        display_name="Nucleotide-Level Metrics",
        show_tp_tn_fp_fn=True,
    ),
    "ml_neighborhood_hit_metrics": PlotMetadata(
        display_name="Neighborhood Hit Metrics",
        icon_path=ICON_PATH / "overlap.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_internal_hit_metrics": PlotMetadata(
        display_name="Internal Hit Metrics",
        icon_path=ICON_PATH / "internal.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_full_coverage_hit_metrics": PlotMetadata(
        display_name="Full Coverage Hit Metrics",
        icon_path=ICON_PATH / "full_coverage.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_perfect_boundary_hit_metrics": PlotMetadata(
        display_name="Perfect Boundary Hit Metrics",
        icon_path=ICON_PATH / "prefect_hit.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_inner_section_boundaries_metrics": PlotMetadata(
        display_name="Inner Section Boundaries",
        show_tp_tn_fp_fn=True,
    ),
    "ml_all_section_boundaries_metrics": PlotMetadata(
        display_name="All Section Boundaries",
        show_tp_tn_fp_fn=True,
    ),
    # IoU
    "iou_average": PlotMetadata(
        display_name="Average IoU",
        icon_path=ICON_PATH / "iou.png",
        description="Measures the intersection over the union of any 2"
                    " overlapping ground truth and predicted section.",
    ),
    "iou_distribution": PlotMetadata(
        display_name="IoU Distribution",
        icon_path=ICON_PATH / "iou.png",
        description="Measures the intersection over the union of any 2"
                    " overlapping ground truth and predicted section.",
    ),
    # Frameshift
    "frameshift": PlotMetadata(display_name="Frameshift Distribution"),
}