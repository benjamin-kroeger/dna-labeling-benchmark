"""Vector (matplotlib-native) drawings of the eight INDEL event icons.

Each ``draw_*`` function renders into an axes that already has the canonical
coordinate frame applied by ``_setup_icon_ax``:  xlim 0–10, ylim 0–5,
GT track centred at y=3.75, Pred track centred at y=1.25.

The public dispatch table is ``VECTOR_ICON_DRAWERS``:
    {event_type_str: draw_fn(ax) -> None}
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

_GT_COLOR   = "#4B3E94"   # purple — ground truth
_PRED_COLOR = "#E8820A"   # orange — prediction
_LW         = 2.0         # uniform line / edge width

_GT_Y  = 3.75   # GT row centre (data units)
_PR_Y  = 1.25   # Pred row centre (data units)
_BH    = 0.70   # box half-height → box spans [y - _BH, y + _BH]
_R     = 0.25   # FancyBboxPatch corner radius (data units)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _setup_icon_ax(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("auto")
    ax.axis("off")


def _intron(ax: plt.Axes, y: float, x0: float, x1: float, color: str) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=_LW, solid_capstyle="butt", zorder=1)


def _exon(ax: plt.Axes, x0: float, x1: float, y: float, color: str) -> None:
    w = x1 - x0
    h = 2 * _BH
    box = mpatches.FancyBboxPatch(
        (x0, y - _BH), w, h,
        boxstyle=f"round,pad=0,rounding_size={_R}",
        linewidth=_LW,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(box)


def _track(ax: plt.Axes, y: float, x0: float, x1: float, color: str,
           x_start: float = 0.0, x_end: float = 10.0) -> None:
    """Draw a complete intron–exon–intron track."""
    _intron(ax, y, x_start, x0, color)
    _exon(ax, x0, x1, y, color)
    _intron(ax, y, x1, x_end, color)


# ---------------------------------------------------------------------------
# The eight event icons
# ---------------------------------------------------------------------------

def draw_5_prime_extensions(ax: plt.Axes) -> None:
    """Pred left edge extends past GT left edge."""
    _setup_icon_ax(ax)
    _track(ax, _GT_Y, x0=3.5, x1=8.5, color=_GT_COLOR)
    _track(ax, _PR_Y, x0=1.5, x1=8.5, color=_PRED_COLOR)


def draw_3_prime_extensions(ax: plt.Axes) -> None:
    """Pred right edge extends past GT right edge."""
    _setup_icon_ax(ax)
    _track(ax, _GT_Y, x0=2.5, x1=7.5, color=_GT_COLOR)
    _track(ax, _PR_Y, x0=2.5, x1=9.5, color=_PRED_COLOR)


def draw_whole_insertions(ax: plt.Axes) -> None:
    """Pred has an exon where GT has only an intron."""
    _setup_icon_ax(ax)
    _intron(ax, _GT_Y, 0, 10, _GT_COLOR)                 # GT: bare intron line
    _track(ax, _PR_Y, x0=3.0, x1=8.0, color=_PRED_COLOR)


def draw_joined(ax: plt.Axes) -> None:
    """Pred merges two adjacent GT exons into one."""
    _setup_icon_ax(ax)
    # GT: two exons
    _intron(ax, _GT_Y, 0, 1.5, _GT_COLOR)
    _exon(ax, 1.5, 4.5, _GT_Y, _GT_COLOR)
    _intron(ax, _GT_Y, 4.5, 5.5, _GT_COLOR)
    _exon(ax, 5.5, 8.5, _GT_Y, _GT_COLOR)
    _intron(ax, _GT_Y, 8.5, 10, _GT_COLOR)
    # Pred: one large merged exon
    _track(ax, _PR_Y, x0=1.5, x1=8.5, color=_PRED_COLOR)


def draw_5_prime_deletions(ax: plt.Axes) -> None:
    """Pred left edge is inside GT (5′ end truncated)."""
    _setup_icon_ax(ax)
    _track(ax, _GT_Y, x0=2.5, x1=8.5, color=_GT_COLOR)
    _track(ax, _PR_Y, x0=4.5, x1=8.5, color=_PRED_COLOR)


def draw_3_prime_deletions(ax: plt.Axes) -> None:
    """Pred right edge is inside GT (3′ end truncated)."""
    _setup_icon_ax(ax)
    _track(ax, _GT_Y, x0=2.5, x1=8.5, color=_GT_COLOR)
    _track(ax, _PR_Y, x0=2.5, x1=6.0, color=_PRED_COLOR)


def draw_split(ax: plt.Axes) -> None:
    """Pred splits one GT exon into two."""
    _setup_icon_ax(ax)
    # GT: one large exon
    _track(ax, _GT_Y, x0=1.5, x1=8.5, color=_GT_COLOR)
    # Pred: two exons
    _intron(ax, _PR_Y, 0, 1.5, _PRED_COLOR)
    _exon(ax, 1.5, 4.5, _PR_Y, _PRED_COLOR)
    _intron(ax, _PR_Y, 4.5, 5.5, _PRED_COLOR)
    _exon(ax, 5.5, 8.5, _PR_Y, _PRED_COLOR)
    _intron(ax, _PR_Y, 8.5, 10, _PRED_COLOR)


def draw_whole_deletions(ax: plt.Axes) -> None:
    """GT has an exon; pred has only an intron (exon missed entirely)."""
    _setup_icon_ax(ax)
    _track(ax, _GT_Y, x0=2.5, x1=8.5, color=_GT_COLOR)
    _intron(ax, _PR_Y, 0, 10, _PRED_COLOR)               # Pred: bare intron line


# ---------------------------------------------------------------------------
# Public dispatch table
# ---------------------------------------------------------------------------

VECTOR_ICON_DRAWERS: dict[str, object] = {
    "5_prime_extensions":  draw_5_prime_extensions,
    "3_prime_extensions":  draw_3_prime_extensions,
    "whole_insertions":    draw_whole_insertions,
    "joined":              draw_joined,
    "5_prime_deletions":   draw_5_prime_deletions,
    "3_prime_deletions":   draw_3_prime_deletions,
    "split":               draw_split,
    "whole_deletions":     draw_whole_deletions,
}
