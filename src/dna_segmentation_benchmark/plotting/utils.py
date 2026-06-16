import textwrap
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dna_segmentation_benchmark.plotting.config import PlotMetadata

SPEZI_COLORS = [
    "#F47B20", "#F5A623", "#F7C94B", "#F0553A", "#E8006A",
    "#F0399A", "#F472B6", "#C0145A", "#7B2D8B", "#4B1A8C",
]


def spezi_palette(n: int | None = None) -> list[tuple[float, float, float]]:
    """Return the spezi palette reordered for max contrast between adjacent colors.

    Picks alternately from the front and back of the gradient so neighbouring
    colors land on opposite ends of the warm→cool spectrum.
    """
    lo, hi = 0, len(SPEZI_COLORS) - 1
    ordered: list[str] = []
    while lo <= hi:
        ordered.append(SPEZI_COLORS[lo])
        lo += 1
        if lo <= hi:
            ordered.append(SPEZI_COLORS[hi])
            hi -= 1
    if n is not None:
        ordered = ordered[:n]
    return sns.color_palette(ordered)


def severity_palette(n: int) -> list[tuple[float, float, float]]:
    """Return *n* colours along a green→red ramp (best → worst).

    Index 0 is green (good), index ``n-1`` is red (bad). Use for ordinal
    categories such as the transcript match classes, so a stacked bar reads as
    a quality gradient rather than arbitrary categorical hues.
    """
    cmap = plt.get_cmap("RdYlGn_r")
    if n <= 1:
        return [tuple(cmap(0.0)[:3])]
    return [tuple(cmap(x)[:3]) for x in np.linspace(0.0, 1.0, n)]


def text_color_for_bg(rgb: tuple[float, float, float]) -> str:
    """Pick ``"black"`` or ``"white"`` for legible text on background *rgb*.

    Uses relative luminance so annotations stay readable on both light and
    dark bar sections.
    """
    r, g, b = rgb[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.6 else "white"


def _annotate_bar_with_errorbar(
    ax: plt.Axes,
    patch,
    height: float,
    se: float | None,
    fmt: str = "%.3f",
    text_fontsize: float = 9.0,
) -> None:
    """Draw an error bar and smart-positioned value label for one bar patch.

    Label placement rule (in rendered pixels):
      - error bar height < 2 text lines → label above the error bar cap
      - error bar height >= 2 text lines → label inside the bar, slightly left,
        rotated 90° so it doesn't overlap the error bar
    When se is None the label is placed above the bar with standard padding.
    """
    bar_x = patch.get_x() + patch.get_width() / 2
    label_text = fmt % height

    if se is not None and height > 0:
        ax.errorbar(bar_x, height, yerr=se, fmt="none", color="black", capsize=3, linewidth=1, zorder=5)

        _, pix_bar = ax.transData.transform((bar_x, height))
        _, pix_top = ax.transData.transform((bar_x, height + se))
        errbar_px = pix_top - pix_bar
        two_lines_px = 2.0 * text_fontsize * ax.figure.dpi / 72.0

        if errbar_px < two_lines_px:
            ax.text(bar_x, height + se, label_text, ha="center", va="bottom", fontsize=text_fontsize)
        else:
            ax.text(
                bar_x - patch.get_width() * 0.15,
                height * 0.5,
                label_text,
                ha="center", va="center",
                fontsize=text_fontsize,
                rotation=90,
            )
    else:
        ax.text(bar_x, height, label_text, ha="center", va="bottom", fontsize=text_fontsize)


def _save_figure(fig: plt.Figure, save_path: Path, logger, dpi: int = 300) -> None:
    """Save *fig* to *save_path*, creating parent dirs as needed."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    logger.info("Saved figure to %s", save_path)


def _composite_on_white(img: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA float32 image against a white background → RGB.

    Resolves the alpha channel before matplotlib ever sees the data, so
    semi-transparent edge pixels (antialiased lines in icons) are always
    blended against white rather than against whatever colour happens to be
    behind the rendered artist.
    """
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32)
        return rgb * alpha + (1.0 - alpha)  # white = 1.0 in float space
    return img


def _add_icon_to_ax(
    ax: plt.Axes,
    event_type: str,
    logger,
    y_rel_pos: float = 1.25,
    icon_height_frac: float = 0.38,
) -> None:
    """Draw a vector icon above *ax* in a small inset axes.

    ``y_rel_pos`` is the vertical centre of the icon in axes-fraction
    coordinates (>1 = above the axes).  ``icon_height_frac`` controls the
    inset height as a fraction of the parent axes height.
    """
    from .icons import VECTOR_ICON_DRAWERS
    draw_fn = VECTOR_ICON_DRAWERS.get(event_type)
    if draw_fn is None:
        logger.warning("No vector icon for event type: %s", event_type)
        return

    y0 = y_rel_pos - icon_height_frac / 2
    icon_ax = ax.inset_axes([0.0, y0, 1.0, icon_height_frac])
    icon_ax.set_in_layout(False)
    icon_ax.set_clip_on(False)
    icon_ax.patch.set_visible(False)
    for spine in icon_ax.spines.values():
        spine.set_visible(False)
    draw_fn(icon_ax)


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

    # --- Icon(s): a single icon, or a vertical stack of two -----------------
    # Two pictograms are stacked (one above the other) rather than placed in a
    # row, so each stays wide enough to read in the narrow panel. Single-icon
    # plots are unchanged.
    max_icon_area_frac = 0.40  # vertical budget for the whole icon area
    icon_paths = [p for p in (metadata.icon_path, metadata.secondary_icon_path) if p is not None]
    if icon_paths:
        n_icons = len(icon_paths)
        per_icon_width_frac = 0.85 if n_icons == 1 else 0.62
        per_icon_h_in = (max_icon_area_frac * panel_height_in) / n_icons
        stack_gap_frac = (0.0 if n_icons == 1 else 0.02) * panel_h
        try:
            # (composited img, w_px, h_px, w_fig_frac, h_fig_frac) per icon
            rendered = []
            for icon_path in icon_paths:
                icon_img = plt.imread(str(icon_path))
                icon_w_px = icon_img.shape[1]
                icon_h_px = icon_img.shape[0]

                # Zoom to fit within the per-icon width budget and the per-icon
                # height budget (the latter shrinks as icons are stacked).
                zoom_w = (panel_width_in * fig.dpi * per_icon_width_frac) / icon_w_px
                zoom_h = (per_icon_h_in * fig.dpi) / icon_h_px
                zoom = min(zoom_w, zoom_h)

                w_fig_frac = ((icon_w_px * zoom) / fig.dpi) / fig_w_in
                h_fig_frac = ((icon_h_px * zoom) / fig.dpi) / fig_h_in
                rendered.append((icon_img, icon_w_px, icon_h_px, w_fig_frac, h_fig_frac))

            # Stack top-down, each icon centered horizontally.
            y = y_cursor
            for icon_img, icon_w_px, icon_h_px, w_fig_frac, h_fig_frac in rendered:
                x = panel_x0 + (panel_w - w_fig_frac) / 2
                icon_ax = fig.add_axes([x, y - h_fig_frac, w_fig_frac, h_fig_frac])
                icon_ax.set_in_layout(False)
                icon_ax.imshow(_composite_on_white(icon_img), interpolation="lanczos")
                # Pad limits slightly so edge pixels are never clipped
                icon_ax.set_xlim(-1, icon_w_px)
                icon_ax.set_ylim(icon_h_px, -3)
                icon_ax.set_axis_off()
                y -= h_fig_frac + stack_gap_frac

            y_cursor = y - panel_h * 0.02  # spacing after the icon area
        except Exception:
            logger.warning(
                "Could not load panel icon(s): %s",
                icon_paths,
                exc_info=True,
            )

    # --- Adaptive text flow -------------------------------------------------
    # Everything below the icons is laid out as one top-down flow that is fitted
    # to the available panel height: blocks are vertically justified (with a
    # capped gap) when there is room, and uniformly shrunk when there is not, so
    # panels never clip bullets off the bottom nor leave large dead gaps.
    wrap_width = 30
    text_x_left = panel_x0 + panel_w * 0.05
    pt_to_frac = 1.0 / 72.0 / fig_h_in  # one typographic point in figure fraction

    def _block_height(n_lines: int, fontsize: float, linespacing: float, boxed: bool) -> float:
        h = n_lines * fontsize * linespacing * pt_to_frac
        if boxed:
            h += 2 * 0.45 * fontsize * pt_to_frac  # round-box top + bottom padding
        return h

    # Build the flowing blocks (description \u2192 bullets \u2192 caveat).
    flow: list[dict] = []
    if metadata.description:
        wrapped = textwrap.fill(metadata.description, width=wrap_width)
        flow.append(dict(text=wrapped, fontsize=11.0, linespacing=1.4, style="italic",
                         color="black", bbox=None, n_lines=wrapped.count("\n") + 1))
    if metadata.bullet_points:
        for bullet in metadata.bullet_points:
            bullet_text = textwrap.fill(f"\u2022 {bullet}", width=wrap_width + 4, subsequent_indent="  ")
            flow.append(dict(text=bullet_text, fontsize=10.0, linespacing=1.3, style="normal",
                             color="black", bbox=None, n_lines=bullet_text.count("\n") + 1))
    if metadata.caveat:
        caveat_text = textwrap.fill(f"\u26a0 {metadata.caveat}", width=wrap_width + 4, subsequent_indent="  ")
        flow.append(dict(text=caveat_text, fontsize=8.0, linespacing=1.3, style="normal",
                         color="#8B6914", n_lines=caveat_text.count("\n") + 1,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8DC",
                                   edgecolor="#DAA520", alpha=0.9)))

    # The TP/TN/FP/FN block is anchored at the bottom of the panel; reserve its
    # height so the flowing blocks above stop just short of it.
    tp_render = None
    text_floor = panel_y0 + panel_h * 0.03
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
        tp_fontsize = 8.0
        tp_height = _block_height(definitions.count("\n") + 1, tp_fontsize, 1.5, boxed=True)
        tp_top = panel_y0 + panel_h * 0.05 + tp_height
        text_floor = tp_top + panel_h * 0.02
        tp_render = (definitions, tp_top, tp_fontsize)

    # Fit the flow into [text_floor, y_cursor] and render it top-down.
    if flow:
        available = y_cursor - text_floor
        sum_h = sum(_block_height(b["n_lines"], b["fontsize"], b["linespacing"], b["bbox"] is not None)
                    for b in flow)
        n_gaps = len(flow)  # one gap above each block
        min_gap = 8.0 * pt_to_frac
        max_gap = 40.0 * pt_to_frac
        if sum_h + n_gaps * min_gap > available and sum_h > 0:
            scale = max(0.5, available / (sum_h + n_gaps * min_gap))
            gap = min_gap * scale
        else:
            scale = 1.0
            gap = min(max_gap, (available - sum_h) / n_gaps)

        y = y_cursor
        for b in flow:
            y -= gap
            fontsize = b["fontsize"] * scale
            kwargs = dict(ha="left", va="top", fontsize=fontsize,
                          linespacing=b["linespacing"], style=b["style"], color=b["color"])
            if b["bbox"]:
                kwargs["bbox"] = b["bbox"]
            fig.text(text_x_left, y, b["text"], **kwargs)
            y -= _block_height(b["n_lines"], fontsize, b["linespacing"], b["bbox"] is not None)

    # --- TP / TN / FP / FN definitions (anchored at the bottom of the panel) ---
    if tp_render is not None:
        definitions, tp_top, tp_fontsize = tp_render
        fig.text(
            text_x_left,
            tp_top,
            definitions,
            ha="left",
            va="top",
            fontsize=tp_fontsize,
            linespacing=1.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )
