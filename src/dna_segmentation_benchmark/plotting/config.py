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
        show_tp_tn_fp_fn=True,
        tp_definition="Nucleotides correctly predicted as the target class",
        tn_definition="Nucleotides correctly predicted as NOT the target class",
        fp_definition="Nucleotides incorrectly predicted as the target class",
        fn_definition="Nucleotides incorrectly predicted as NOT the target class",
    ),
    "neighborhood_hit": PlotMetadata(
        display_name="Neighborhood Hit Metrics",
        icon_path=ICON_PATH / "overlap.png",
        show_tp_tn_fp_fn=True,
        description="Measures how well predicted sections partially overlap ground truth section",
        tp_definition="Predicted section overlaps a true section",
        tn_definition="N/A",
        fp_definition="Predicted section does not overlap any ground truth section",
        fn_definition="Ground Truth section is not overlapped by any prediction",
    ),
    "internal_hit": PlotMetadata(
        display_name="Internal Hit Metrics",
        icon_path=ICON_PATH / "internal.png",
        description="Measures if a prediction did not exceed the ground truth boundaries (forgiving to under prediction)",
        show_tp_tn_fp_fn=True,
        tp_definition="Predicted section is completely contained within a true section",
        tn_definition="N/A",
        fp_definition="N/A",
        fn_definition="Ground truth section does not completely contain a predicted section",
    ),
    "full_coverage_hit": PlotMetadata(
        display_name="Full Coverage Hit Metrics",
        icon_path=ICON_PATH / "full_coverage.png",
        description="Measures if a prediction contains the entire ground truth section or more (forgiving to over prediction)",
        show_tp_tn_fp_fn=True,
        tp_definition="Ground Truth section is completely covered by a predicted section",
        tn_definition="N/A",
        fp_definition="N/A",
        fn_definition="Ground Truth section is not completely covered by a predicted section",
    ),
    "perfect_boundary_hit": PlotMetadata(
        display_name="Perfect Boundary Hit Metrics",
        icon_path=ICON_PATH / "prefect_hit.png",
        show_tp_tn_fp_fn=True,
        tp_definition="Predicted section exactly matches a true section's boundaries (100% IoU)",
        tn_definition="N/A",
        fp_definition="Predicted section does not perfectly match any true section",
        fn_definition="Ground Truth section is not perfectly matched by a prediction",
    ),
    "inner_section_boundaries": PlotMetadata(
        display_name="Inner Section Boundaries",
        show_tp_tn_fp_fn=True,
        tp_definition="All inner section boundaries are correct except for the outer ones",
        tn_definition="N/A",
        fp_definition="Not all inner section boundaries are correct",
        fn_definition="No predictions were made but gt sections exist",
    ),
    "all_section_boundaries": PlotMetadata(
        display_name="All Section Boundaries",
        show_tp_tn_fp_fn=True,
        tp_definition="All section boundaries are correct",
        tn_definition="N/A",
        fp_definition="Not all section boundaries are correct",
        fn_definition="No predictions were made but gt sections exist",
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