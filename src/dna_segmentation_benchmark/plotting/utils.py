import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from dna_segmentation_benchmark.plotting.config import DEFAULT_FIG_SIZE, PlotMetadata


def _save_figure(fig: plt.Figure, save_path: Path, logger) -> None:
    """Save *fig* to *save_path*, creating parent dirs as needed."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    logger.info("Saved figure to %s", save_path)


def _add_icon_to_ax(
    ax: plt.Axes,
    icon_path: str,
    logger,
    zoom: float = 0.2,
    x_rel_pos: float = 0.5,
    y_rel_pos: float = 1.25,
) -> None:
    """Place an image (icon) above *ax*."""
    try:
        icon_img = plt.imread(icon_path)
        imagebox = OffsetImage(icon_img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x_rel_pos, y_rel_pos), xycoords=ax.transAxes, frameon=False)
        ax.add_artist(ab)
    except FileNotFoundError:
        logger.warning("Icon not found: %s", icon_path)
    except Exception:
        logger.warning("Could not load icon: %s", icon_path, exc_info=True)


def _add_pictogram_panel(
    fig: plt.Figure,
    metadata: PlotMetadata | None,
    logger,
    panel_width_fraction: float = 0.22,
) -> None:
    """Add a right-side pictogram panel to *fig*.

    The panel displays an icon (scaled to fit within the panel) with
    explanatory text below it, and optionally a TP/TN/FP/FN
    definitions block.  All existing axes in *fig* are shrunk to make
    room.

    If *metadata* is ``None`` or contains nothing to render, the
    function is a **no-op**.
    """
    if metadata is None:
        return
    has_content = (
        metadata.icon_path is not None
        or metadata.description
        or metadata.bullet_points
        or metadata.caveat
        or metadata.show_tp_tn_fp_fn
        or metadata.display_name
    )
    if not has_content:
        return

    # This helper uses manual figure coordinates.  If a caller created the
    # figure with constrained_layout=True, matplotlib can later override the
    # positions below during draw/savefig and place plot axes under the panel.
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    else:
        fig.set_constrained_layout(False)

    existing_axes = fig.get_axes()
    if not existing_axes:
        return

    # Rescale the whole existing axes layout into the left content area.
    # Multiplying each axes width alone is not enough for multi-panel figures:
    # a right-hand subplot can still extend into the panel because its x0 is
    # already far to the right.
    left_edge = min(ax.get_position().x0 for ax in existing_axes)
    right_edge = max(ax.get_position().x1 for ax in existing_axes)
    content_right = 1 - panel_width_fraction - 0.03
    scale = (content_right - left_edge) / (right_edge - left_edge)

    for ax in existing_axes:
        box = ax.get_position()
        ax.set_position(
            [
                left_edge + (box.x0 - left_edge) * scale,
                box.y0,
                box.width * scale,
                box.height,
            ]
        )

    # Create the panel axes on the freed right-hand side
    panel_left = 1 - panel_width_fraction + 0.01
    panel_width = panel_width_fraction - 0.02
    panel_ax = fig.add_axes([panel_left, 0.05, panel_width, 0.90])
    panel_ax.set_in_layout(False)
    panel_ax.set_axis_off()

    # Panel dimensions in inches
    fig_w_in, fig_h_in = fig.get_size_inches()
    panel_width_in = panel_width * fig_w_in
    panel_height_in = 0.90 * fig_h_in  # panel is 90% of fig height

    # Convert panel_ax coordinates to figure coordinates for precise placement
    # of elements relative to the panel.
    # panel_ax.transAxes.transform((x, y)) gives figure coordinates.
    # fig.transFigure.inverted().transform((x, y)) gives figure fraction coordinates.
    # We want to work in figure fraction coordinates for text and icon placement.
    panel_bbox = panel_ax.get_position()
    panel_x0, panel_y0, panel_w, panel_h = panel_bbox.x0, panel_bbox.y0, panel_bbox.width, panel_bbox.height

    # y_cursor is in figure fraction coordinates, relative to the top of the panel
    y_cursor = panel_y0 + panel_h * 0.95  # Start near the top of the panel

    # --- Display name ---
    if metadata.display_name:
        # Text x-position is center of panel, y-position is y_cursor
        text_x = panel_x0 + panel_w / 2
        fig.text(
            text_x,
            y_cursor,
            metadata.display_name,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            wrap=True,
        )
        y_cursor -= panel_h * 0.08  # Move cursor down

    # --- Icon (scaled to fit panel, constrained by both width and height) ---
    # Reserve vertical budget: title ~8%, icon max 40%, description ~20%, TP/TN ~20%
    max_icon_height_frac = 0.40  # max 40% of panel height for the icon
    if metadata.icon_path is not None:
        try:
            icon_img = plt.imread(str(metadata.icon_path))
            icon_w_px = icon_img.shape[1]
            icon_h_px = icon_img.shape[0]

            # Calculate zoom to fit within 85% of panel width and max_icon_height_frac of panel height
            zoom_w = (panel_width_in * fig.dpi * 0.85) / icon_w_px
            max_icon_h_px = max_icon_height_frac * panel_height_in * fig.dpi
            zoom_h = max_icon_h_px / icon_h_px
            zoom = min(zoom_w, zoom_h)

            # Create an inset axes for the icon to ensure it respects bounds
            icon_rendered_w_in = (icon_w_px * zoom) / fig.dpi
            icon_rendered_h_in = (icon_h_px * zoom) / fig.dpi

            # Convert rendered dimensions to figure fraction
            icon_w_fig_frac = icon_rendered_w_in / fig_w_in
            icon_h_fig_frac = icon_rendered_h_in / fig_h_in

            # Calculate icon axes position: centered horizontally, top aligned with y_cursor
            icon_ax_x0 = panel_x0 + (panel_w - icon_w_fig_frac) / 2
            icon_ax_y0 = y_cursor - icon_h_fig_frac  # Top of icon is at y_cursor

            icon_ax = fig.add_axes([icon_ax_x0, icon_ax_y0, icon_w_fig_frac, icon_h_fig_frac])
            icon_ax.set_in_layout(False)
            icon_ax.imshow(icon_img)
            # Pad limits slightly so edge pixels are never clipped
            icon_ax.set_xlim(-1, icon_w_px)
            icon_ax.set_ylim(icon_h_px, -3)
            icon_ax.set_axis_off()

            y_cursor -= icon_h_fig_frac + panel_h * 0.04  # Move cursor down past icon and add spacing
        except Exception:
            logger.warning(
                "Could not load panel icon: %s",
                metadata.icon_path,
                exc_info=True,
            )

    # --- Description text ---
    wrap_width = 30
    text_x_left = panel_x0 + panel_w * 0.05
    line_step = panel_h * 0.04

    if metadata.description:
        wrapped = textwrap.fill(metadata.description, width=wrap_width)
        fig.text(
            text_x_left,
            y_cursor,
            wrapped,
            ha="left",
            va="top",
            fontsize=11,
            linespacing=1.4,
            style="italic",
        )
        n_lines = wrapped.count("\n") + 1
        y_cursor -= n_lines * line_step + panel_h * 0.03

    # --- Bullet points ---
    if metadata.bullet_points:
        for bullet in metadata.bullet_points:
            bullet_text = textwrap.fill(
                f"\u2022 {bullet}",
                width=wrap_width + 4,
                subsequent_indent="  ",
            )
            fig.text(
                text_x_left,
                y_cursor,
                bullet_text,
                ha="left",
                va="top",
                fontsize=10,
                linespacing=1.3,
            )
            n_lines = bullet_text.count("\n") + 1
            y_cursor -= n_lines * line_step * 0.85

        y_cursor -= panel_h * 0.02  # spacing after bullets

    # --- Caveat box ---
    if metadata.caveat:
        caveat_text = textwrap.fill(
            f"\u26a0 {metadata.caveat}",
            width=wrap_width + 4,
            subsequent_indent="  ",
        )
        fig.text(
            text_x_left,
            y_cursor,
            caveat_text,
            ha="left",
            va="top",
            fontsize=8,
            linespacing=1.3,
            color="#8B6914",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#FFF8DC",
                edgecolor="#DAA520",
                alpha=0.9,
            ),
        )
        n_lines = caveat_text.count("\n") + 1
        y_cursor -= n_lines * line_step + panel_h * 0.03

    # --- TP / TN / FP / FN definitions (placed at bottom of panel) ---
    if metadata.show_tp_tn_fp_fn:
        parts = []
        for label, attr, default in (
            ("TP", "tp_definition", "Correctly predicted"),
            ("TN", "tn_definition", "Correctly absent"),
            ("FP", "fp_definition", "Falsely predicted"),
            ("FN", "fn_definition", "Falsely missed"),
        ):
            defn = getattr(metadata, attr, None) or default
            parts.append(f"\u2022 {label}: {defn}")

        wrapped_parts = [textwrap.fill(p, width=wrap_width + 8, subsequent_indent="  ") for p in parts]
        definitions = "\n".join(wrapped_parts)

        # Place at fixed position near the bottom to avoid overlap
        tp_y_bottom = panel_y0 + panel_h * 0.05
        estimated_line_height = 0.025 * (fig_h_in / DEFAULT_FIG_SIZE[1])
        tp_y_top = tp_y_bottom + 4 * estimated_line_height * 1.5
        final_tp_y = min(y_cursor - panel_h * 0.02, tp_y_top)

        fig.text(
            text_x_left,
            final_tp_y,
            definitions,
            ha="left",
            va="top",
            fontsize=8,
            linespacing=1.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )
