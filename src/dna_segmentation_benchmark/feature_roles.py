"""Feature-role mapping helpers for file-based annotation parsing.

Stage 2 introduces explicit feature-role maps so GFF/GTF feature names are
translated into benchmark semantics before arrays are painted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .label_definition import AnnotationMode, BenchmarkScope, LabelConfig

FeatureRoleMap = dict[str, str]
PredFeatureRoleMapInput = FeatureRoleMap | dict[str, FeatureRoleMap] | None

EXON_ROLE = "exon"
FIVE_PRIME_UTR_ROLE = "five_prime_utr"
CDS_ROLE = "cds"
THREE_PRIME_UTR_ROLE = "three_prime_utr"
INTRON_ROLE = "intron"
SPLICE_DONOR_ROLE = "splice_donor"
SPLICE_ACCEPTOR_ROLE = "splice_acceptor"

_TRANSCRIBED_ROLES = frozenset({
    EXON_ROLE,
    FIVE_PRIME_UTR_ROLE,
    CDS_ROLE,
    THREE_PRIME_UTR_ROLE,
})


def default_feature_role_map(label_config: LabelConfig) -> FeatureRoleMap:
    """Return the default feature-role map for the selected annotation mode."""
    if label_config.annotation_mode == AnnotationMode.EXON_INTRON:
        return {
            "exon": EXON_ROLE,
            "CDS": EXON_ROLE,
        }
    return {
        "five_prime_UTR": FIVE_PRIME_UTR_ROLE,
        "five_prime_utr": FIVE_PRIME_UTR_ROLE,
        "CDS": CDS_ROLE,
        "three_prime_UTR": THREE_PRIME_UTR_ROLE,
        "three_prime_utr": THREE_PRIME_UTR_ROLE,
    }


def allowed_feature_roles(label_config: LabelConfig) -> frozenset[str]:
    """Return the semantic feature roles supported by *label_config*."""
    if label_config.annotation_mode == AnnotationMode.EXON_INTRON:
        roles = {EXON_ROLE}
    else:
        roles = {FIVE_PRIME_UTR_ROLE, CDS_ROLE, THREE_PRIME_UTR_ROLE}
    if label_config.intron_label is not None:
        roles.add(INTRON_ROLE)
    if label_config.splice_donor_label is not None:
        roles.add(SPLICE_DONOR_ROLE)
        roles.add(SPLICE_ACCEPTOR_ROLE)
    return frozenset(roles)


def role_to_label(label_config: LabelConfig, role: str) -> int:
    """Resolve a semantic role string to the configured integer label."""
    if role == EXON_ROLE:
        if label_config.exon_label is None:
            raise ValueError("Role 'exon' is not available for this LabelConfig.")
        return label_config.exon_label
    if role == FIVE_PRIME_UTR_ROLE:
        if label_config.five_prime_utr_label is None:
            raise ValueError("Role 'five_prime_utr' is not available for this LabelConfig.")
        return label_config.five_prime_utr_label
    if role == CDS_ROLE:
        if label_config.cds_label is None:
            raise ValueError("Role 'cds' is not available for this LabelConfig.")
        return label_config.cds_label
    if role == THREE_PRIME_UTR_ROLE:
        if label_config.three_prime_utr_label is None:
            raise ValueError("Role 'three_prime_utr' is not available for this LabelConfig.")
        return label_config.three_prime_utr_label
    if role == INTRON_ROLE:
        if label_config.intron_label is None:
            raise ValueError("Role 'intron' is not available for this LabelConfig.")
        return label_config.intron_label
    if role == SPLICE_DONOR_ROLE:
        if label_config.splice_donor_label is None:
            raise ValueError("Role 'splice_donor' is not available for this LabelConfig.")
        return label_config.splice_donor_label
    if role == SPLICE_ACCEPTOR_ROLE:
        if label_config.splice_acceptor_label is None:
            raise ValueError("Role 'splice_acceptor' is not available for this LabelConfig.")
        return label_config.splice_acceptor_label
    raise ValueError(f"Unknown feature role: {role!r}")


def normalize_feature_role_map(
    feature_role_map: Mapping[str, str] | None,
    label_config: LabelConfig,
    *,
    arg_name: str,
) -> FeatureRoleMap:
    """Validate and normalize one feature-role map."""
    raw_map = default_feature_role_map(label_config) if feature_role_map is None else dict(feature_role_map)
    if not raw_map:
        raise ValueError(f"{arg_name} must not be empty.")

    allowed_roles = allowed_feature_roles(label_config)
    normalized: FeatureRoleMap = {}
    for feature_type, role in raw_map.items():
        if not isinstance(feature_type, str) or not feature_type:
            raise ValueError(f"{arg_name} keys must be non-empty strings.")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{arg_name}[{feature_type!r}] must map to a non-empty role string.")
        if role not in allowed_roles:
            raise ValueError(
                f"{arg_name}[{feature_type!r}]={role!r} is not valid for annotation_mode="
                f"{label_config.annotation_mode.value}. Allowed roles: {sorted(allowed_roles)}"
            )
        # Force early validation that the target label exists.
        role_to_label(label_config, role)
        normalized[feature_type] = role
    return normalized


def normalize_pred_feature_role_maps(
    pred_names: list[str],
    pred_feature_role_maps: PredFeatureRoleMapInput,
    *,
    default: FeatureRoleMap,
    label_config: LabelConfig,
) -> dict[str, FeatureRoleMap]:
    """Return one validated feature-role map per predictor."""
    if pred_feature_role_maps is None:
        return {
            name: normalize_feature_role_map(default, label_config, arg_name=f"pred_feature_role_maps[{name!r}]")
            for name in pred_names
        }

    if isinstance(pred_feature_role_maps, Mapping) and pred_feature_role_maps:
        first_value = next(iter(pred_feature_role_maps.values()))
        if isinstance(first_value, Mapping):
            return {
                name: normalize_feature_role_map(
                    pred_feature_role_maps.get(name, default),
                    label_config,
                    arg_name=f"pred_feature_role_maps[{name!r}]",
                )
                for name in pred_names
            }

    return {
        name: normalize_feature_role_map(
            pred_feature_role_maps,
            label_config,
            arg_name=f"pred_feature_role_maps[{name!r}]",
        )
        for name in pred_names
    }


def exonic_feature_types(feature_role_map: Mapping[str, str]) -> list[str]:
    """Return feature types that contribute to the transcript-exon span."""
    return [feature_type for feature_type, role in feature_role_map.items() if role in _TRANSCRIBED_ROLES]


def feature_types_for_scope(
    feature_role_map: Mapping[str, str],
    label_config: LabelConfig,
    scope: BenchmarkScope | str,
) -> list[str]:
    """Return feature types whose mapped labels belong to *scope*."""
    scope_tokens = label_config.scope_tokens(scope)
    return [
        feature_type
        for feature_type, role in feature_role_map.items()
        if role_to_label(label_config, role) in scope_tokens
    ]


def feature_role_paint_plan(
    feature_role_map: Mapping[str, str],
    label_config: LabelConfig,
) -> list[tuple[int, list[str]]]:
    """Return ordered ``(label_value, feature_types)`` paint passes."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for feature_type, role in feature_role_map.items():
        grouped[role].append(feature_type)

    if label_config.annotation_mode == AnnotationMode.EXON_INTRON:
        ordered_roles = [INTRON_ROLE, EXON_ROLE, SPLICE_DONOR_ROLE, SPLICE_ACCEPTOR_ROLE]
    else:
        ordered_roles = [
            INTRON_ROLE,
            FIVE_PRIME_UTR_ROLE,
            CDS_ROLE,
            THREE_PRIME_UTR_ROLE,
            SPLICE_DONOR_ROLE,
            SPLICE_ACCEPTOR_ROLE,
        ]

    plan: list[tuple[int, list[str]]] = []
    for role in ordered_roles:
        feature_types = grouped.get(role)
        if not feature_types:
            continue
        plan.append((role_to_label(label_config, role), feature_types))
    return plan


def paint_feature_rows(
    arr: np.ndarray,
    features_df: pd.DataFrame,
    region_start: int,
    feature_role_map: Mapping[str, str],
    label_config: LabelConfig,
) -> None:
    """Paint feature rows into *arr* according to the configured role map."""
    for label_value, feature_types in feature_role_paint_plan(feature_role_map, label_config):
        subset = features_df[features_df["type"].isin(feature_types)]
        for feat_start, feat_end in zip(subset["start"], subset["end"]):
            if pd.isna(feat_start) or pd.isna(feat_end):
                continue
            local_start = max(0, int(feat_start) - region_start)
            local_end = min(len(arr), int(feat_end) - region_start + 1)
            if local_start < local_end:
                arr[local_start:local_end] = label_value
