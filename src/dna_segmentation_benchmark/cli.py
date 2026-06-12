"""Command-line interface for the DNA segmentation benchmark.

Usage examples::

    # Basic run with a single prediction
    dna-benchmark run \\
        --gt ground_truth.gff \\
        --pred augustus:predictions.gff \\
        --config label_config.yaml

    # Augustus (CDS features) with BEST_PER_LOCUS matching
    dna-benchmark run \\
        --gt ground_truth.gff \\
        --pred augustus:augustus.gff \\
        --config label_config.yaml \\
        --locus-matching best_per_locus

    # Compare multiple predictors at once
    dna-benchmark run \\
        --gt ground_truth.gff \\
        --pred augustus:augustus.gff \\
        --pred helixer:helixer.gff \\
        --config label_config.yaml \\
        --mapping-output mapping_debug.tsv \\
        --output results.json

    # Generate a template config file
    dna-benchmark init-config --output my_config.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from .eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_multiple,
)
from .eval.global_metrics import compute_global_metrics
from .label_definition import LabelConfig
from .transcript_mapping import LocusMatchingMode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_METRIC_NAMES = {m.name: m for m in EvalMetrics if not m.name.startswith("_")}
_MODE_NAMES = {m.value: m for m in LocusMatchingMode}


def _parse_pred_spec(spec: str) -> tuple[str, Path]:
    """Parse a ``name:path`` prediction specifier.

    If no colon is present, the file stem is used as the predictor name.

    Parameters
    ----------
    spec : str
        Either ``"name:/path/to/file.gff"`` or ``"/path/to/file.gff"``.

    Returns
    -------
    tuple[str, Path]
        ``(predictor_name, file_path)``
    """
    if ":" in spec and not spec.startswith("/"):
        name, _, path_str = spec.partition(":")
        return name, Path(path_str)

    path = Path(spec)
    return path.stem, path


def _load_label_config(path: Path) -> LabelConfig:
    """Load a :class:`LabelConfig` from a YAML file.

    The YAML must declare an ``annotation_mode`` and the labels required by
    that mode.  Two shapes are supported:

    ``EXON_INTRON``::

        annotation_mode: EXON_INTRON
        background_label: 8
        exon_label: 0
        intron_label: 2
        splice_donor_label: 1
        splice_acceptor_label: 3

    ``UTR_CDS_INTRON``::

        annotation_mode: UTR_CDS_INTRON
        background_label: 8
        cds_label: 0
        five_prime_utr_label: 4
        three_prime_utr_label: 5
        intron_label: 2
        splice_donor_label: 1
        splice_acceptor_label: 3

    GFF/GTF feature names such as ``exon``/``CDS`` are CLI options
    (``--gt-exon-feature-type`` / ``--gt-feature-role``), not label config.
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return LabelConfig(**raw)


def _load_npz_to_list(path: Path) -> list[np.ndarray]:
    """Load an ``.npz`` file and return its arrays as a list."""
    npz = np.load(str(path), allow_pickle=True)
    return [npz[key] for key in npz.keys()]


def _parse_pred_exon_feature_specs(
    specs: tuple[str, ...],
) -> str | list[str] | dict[str, str | list[str]] | None:
    """Parse ``--pred-exon-feature-type`` values from the CLI.

    Plain values apply to every predictor, e.g. ``CDS`` or repeated
    ``exon --pred-exon-feature-type CDS``.  Named values use
    ``predictor:type`` and allow mixed predictor inputs, e.g.
    ``augustus:CDS`` and ``helixer:exon``.
    """
    if not specs:
        return None

    has_named = any(":" in spec for spec in specs)
    has_plain = any(":" not in spec for spec in specs)
    if has_named and has_plain:
        raise click.BadParameter(
            "Use either plain feature types for all predictors or predictor:type entries, not both.",
            param_hint="--pred-exon-feature-type",
        )

    if has_plain:
        return list(specs)

    by_predictor: dict[str, list[str]] = {}
    for spec in specs:
        predictor, _, feature_type = spec.partition(":")
        if not predictor or not feature_type:
            raise click.BadParameter(
                "Expected predictor:type, e.g. augustus:CDS.",
                param_hint="--pred-exon-feature-type",
            )
        by_predictor.setdefault(predictor, []).append(feature_type)

    return {predictor: values[0] if len(values) == 1 else values for predictor, values in by_predictor.items()}


def _parse_role_pair(spec: str, *, param_hint: str) -> tuple[str, str]:
    """Parse one ``feature_type:role`` token into a pair."""
    feature_type, sep, role = spec.partition(":")
    if not sep or not feature_type or not role:
        raise click.BadParameter(
            "Expected feature_type:role, e.g. CDS:cds.",
            param_hint=param_hint,
        )
    return feature_type, role


def _parse_gt_feature_role_specs(specs: tuple[str, ...]) -> dict[str, str] | None:
    """Parse ``--gt-feature-role`` values into a feature-role map.

    Each value is ``feature_type:role`` (e.g. ``CDS:cds``).  Returns ``None``
    when no values are given, so the caller can fall back to the legacy
    ``--gt-exon-feature-type`` path.
    """
    if not specs:
        return None
    role_map: dict[str, str] = {}
    for spec in specs:
        feature_type, role = _parse_role_pair(spec, param_hint="--gt-feature-role")
        role_map[feature_type] = role
    return role_map


def _parse_pred_feature_role_specs(
    specs: tuple[str, ...],
) -> dict[str, str] | dict[str, dict[str, str]] | None:
    """Parse ``--pred-feature-role`` values from the CLI.

    Plain ``feature_type:role`` values apply to every predictor.  Named
    ``predictor=feature_type:role`` values build per-predictor maps.  The two
    forms cannot be mixed.  Returns ``None`` when no values are given.
    """
    if not specs:
        return None

    has_named = any("=" in spec for spec in specs)
    has_plain = any("=" not in spec for spec in specs)
    if has_named and has_plain:
        raise click.BadParameter(
            "Use either plain feature_type:role for all predictors or "
            "predictor=feature_type:role entries, not both.",
            param_hint="--pred-feature-role",
        )

    if has_plain:
        role_map: dict[str, str] = {}
        for spec in specs:
            feature_type, role = _parse_role_pair(spec, param_hint="--pred-feature-role")
            role_map[feature_type] = role
        return role_map

    by_predictor: dict[str, dict[str, str]] = {}
    for spec in specs:
        predictor, _, pair = spec.partition("=")
        if not predictor or not pair:
            raise click.BadParameter(
                "Expected predictor=feature_type:role, e.g. helixer=CDS:cds.",
                param_hint="--pred-feature-role",
            )
        feature_type, role = _parse_role_pair(pair, param_hint="--pred-feature-role")
        by_predictor.setdefault(predictor, {})[feature_type] = role
    return by_predictor


def _resolve_pred_exon_types(
    parsed: str | list[str] | dict[str, str | list[str]] | None,
    pred_names: list[str],
    default: list[str],
) -> dict[str, list[str]]:
    """Normalize ``--pred-exon-feature-type`` specs into per-predictor lists."""
    if parsed is None:
        return {name: list(default) for name in pred_names}
    if isinstance(parsed, dict):
        result: dict[str, list[str]] = {}
        for name in pred_names:
            v = parsed.get(name, default)
            result[name] = [v] if isinstance(v, str) else list(v)
        return result
    types = [parsed] if isinstance(parsed, str) else list(parsed)
    return {name: list(types) for name in pred_names}


def _pred_role_map_for(
    pred_role_maps: dict[str, str] | dict[str, dict[str, str]] | None,
    pred_name: str,
) -> dict[str, str] | None:
    """Return the per-predictor feature-role map for *pred_name*, if any.

    A plain ``{feature_type: role}`` map applies to every predictor; a nested
    ``{predictor: {feature_type: role}}`` map is indexed by *pred_name*.
    """
    if pred_role_maps is None:
        return None
    if all(isinstance(value, dict) for value in pred_role_maps.values()):
        return pred_role_maps.get(pred_name)  # type: ignore[return-value]
    return pred_role_maps  # type: ignore[return-value]


def _serialise_results(results: dict) -> dict:
    """Convert numpy types in the results dict to JSON-serialisable types."""
    import pandas as pd

    if isinstance(results, dict):
        return {k: _serialise_results(v) for k, v in results.items()}
    if isinstance(results, (list, tuple)):
        return [_serialise_results(item) for item in results]
    if isinstance(results, np.ndarray):
        return results.tolist()
    if isinstance(results, pd.DataFrame):
        return results.to_dict(orient="records")
    if isinstance(results, pd.Series):
        return results.to_dict()
    if isinstance(results, (np.integer, np.int64)):
        return int(results)
    if isinstance(results, (np.floating, np.float64)):
        return float(results)
    return results


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------


@click.group()
@click.version_option(package_name="dna-segmentation-benchmark")
def cli():
    """DNA Segmentation Benchmark -- evaluate nucleotide-level predictions."""


# ------------------------------------------------------------------
# `run` command
# ------------------------------------------------------------------


@cli.command()
@click.option(
    "--gt",
    "gt_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to ground-truth annotations (.gff3 or .gtf file).",
)
@click.option(
    "--pred",
    "pred_specs",
    required=True,
    type=str,
    multiple=True,
    help=(
        "Prediction file(s).  Format: 'name:path' or just 'path' "
        "(repeatable, e.g. --pred augustus:aug.gff --pred helixer:hel.gff)."
    ),
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to label configuration file (.yaml).",
)
@click.option(
    "--exclude-features",
    "exclude_features",
    type=str,
    multiple=True,
    help="GFF feature types to ignore (e.g. --exclude-features gene).",
)
@click.option(
    "--transcript-types",
    "transcript_types",
    type=str,
    multiple=True,
    default=["mRNA", "transcript"],
    show_default=True,
    help="Feature types that define transcript boundaries (repeatable).",
)
@click.option(
    "--gt-exon-feature-type",
    "gt_exon_feature_types",
    type=str,
    multiple=True,
    default=("exon",),
    show_default=True,
    help=("GT GFF/GTF feature type to treat as exon/coding interval. Repeat for multiple types."),
)
@click.option(
    "--pred-exon-feature-type",
    "pred_exon_feature_specs",
    type=str,
    multiple=True,
    help=(
        "Prediction feature type to treat as exon/coding interval. Repeat plain "
        "values for all predictors, or use predictor:type for mixed inputs "
        "(e.g. augustus:CDS helixer:exon). Defaults to GT feature types."
    ),
)
@click.option(
    "--gt-feature-role",
    "gt_feature_role_specs",
    type=str,
    multiple=True,
    help=(
        "GT feature-role mapping as feature_type:role (repeatable), e.g. "
        "--gt-feature-role CDS:cds --gt-feature-role five_prime_UTR:five_prime_utr. "
        "Required for UTR_CDS_INTRON configs; takes precedence over "
        "--gt-exon-feature-type."
    ),
)
@click.option(
    "--pred-feature-role",
    "pred_feature_role_specs",
    type=str,
    multiple=True,
    help=(
        "Prediction feature-role mapping (repeatable). Use plain "
        "feature_type:role for all predictors, or predictor=feature_type:role "
        "for per-predictor maps (e.g. helixer=CDS:cds). Defaults to the GT "
        "feature-role map."
    ),
)
@click.option(
    "--metrics",
    "metric_names",
    type=click.Choice(list(_METRIC_NAMES.keys()), case_sensitive=False),
    multiple=True,
    default=["REGION_DISCOVERY", "NUCLEOTIDE_CLASSIFICATION"],
    show_default=True,
    help="Metric group(s) to compute (repeatable).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write results to this JSON file instead of stdout.",
)
@click.option(
    "--individual",
    is_flag=True,
    default=False,
    help="Return per-sequence results instead of aggregated.",
)
@click.option(
    "--mapping-output",
    "mapping_output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the GT <-> prediction mapping to this TSV file for debugging.",
)
@click.option(
    "--locus-matching",
    "locus_matching",
    type=click.Choice([m.value for m in LocusMatchingMode], case_sensitive=False),
    default=LocusMatchingMode.FULL_DISCOVERY.value,
    show_default=True,
    help=(
        "Locus matching mode.  'full_discovery' maximises 1:1 matches per locus "
        "(multi-isoform tools).  'best_per_locus' keeps only the single "
        "best-scoring pair per locus (single-transcript tools, e.g. Augustus)."
    ),
)
@click.option(
    "--infer-introns",
    is_flag=True,
    default=False,
    help=(
        "Fill background gaps between adjacent coding segments with the configured intron label before benchmarking."
    ),
)
def run(
    gt_path: Path,
    pred_specs: tuple[str, ...],
    config_path: Path,
    exclude_features: tuple[str, ...],
    transcript_types: tuple[str, ...],
    gt_exon_feature_types: tuple[str, ...],
    pred_exon_feature_specs: tuple[str, ...],
    gt_feature_role_specs: tuple[str, ...],
    pred_feature_role_specs: tuple[str, ...],
    metric_names: tuple[str, ...],
    output_path: Path | None,
    individual: bool,
    mapping_output_path: Path | None,
    locus_matching: str,
    infer_introns: bool,
):
    """Run the benchmark on ground-truth vs. prediction GFF/GTF files."""
    from .io_utils import collect_gff
    from .transcript_mapping import (
        build_paired_arrays,
        export_mapping_table,
        map_transcripts,
    )
    # ------------------------------------------------------------------
    # 1. Parse inputs
    # ------------------------------------------------------------------
    label_config = _load_label_config(config_path)
    mode = _MODE_NAMES[locus_matching.lower()]

    pred_paths: dict[str, Path] = {}
    for spec in pred_specs:
        name, path = _parse_pred_spec(spec)
        if not path.exists():
            raise click.BadParameter(
                f"Prediction file does not exist: {path}",
                param_hint="--pred",
            )
        if name in pred_paths:
            raise click.BadParameter(
                f"Duplicate predictor name '{name}'. Use 'name:path' format to give each prediction a unique name.",
                param_hint="--pred",
            )
        pred_paths[name] = path

    excl_list = list(exclude_features)
    tt_list = list(transcript_types)
    metrics = [_METRIC_NAMES[name.upper()] for name in metric_names]

    # ------------------------------------------------------------------
    # Resolve how GFF/GTF feature names map to label semantics.
    #
    # EXON_INTRON keeps the simple --gt-exon-feature-type / --pred-exon-
    # feature-type path unless explicit feature-role flags are given.
    # UTR_CDS_INTRON always uses feature-role maps (explicit, or the
    # mode default when no flags are passed) because a single exonic
    # feature type cannot express distinct UTR/CDS roles.
    # ------------------------------------------------------------------
    gt_role_map = _parse_gt_feature_role_specs(gt_feature_role_specs)
    pred_role_maps = _parse_pred_feature_role_specs(pred_feature_role_specs)
    is_exon_intron = label_config.annotation_mode.name == "EXON_INTRON"
    use_role_maps = (not is_exon_intron) or gt_role_map is not None or pred_role_maps is not None

    if use_role_maps:
        gt_exon_types: list[str] | None = None
        pred_exon_types_by_name: dict[str, list[str]] | None = None
        gt_map_kwargs = dict(gt_feature_role_map=gt_role_map)
        pred_map_kwargs = dict(pred_feature_role_maps=pred_role_maps)
    else:
        gt_exon_types = list(gt_exon_feature_types)
        pred_exon_feature_types_parsed = _parse_pred_exon_feature_specs(pred_exon_feature_specs)
        pred_exon_types_by_name = _resolve_pred_exon_types(
            pred_exon_feature_types_parsed, list(pred_paths.keys()), gt_exon_types
        )
        # Convert explicit exon-type lists to role maps (new internal API).
        gt_role_map = {ft: "exon" for ft in gt_exon_types}
        pred_role_maps = {name: {ft: "exon" for ft in types} for name, types in pred_exon_types_by_name.items()}
        gt_map_kwargs = dict(gt_feature_role_map=gt_role_map)
        pred_map_kwargs = dict(pred_feature_role_maps=pred_role_maps)

    # ------------------------------------------------------------------
    # 2. Read GFF files once into memory
    # ------------------------------------------------------------------
    click.echo(f"Reading GT: {gt_path.name} and {len(pred_paths)} prediction file(s)...")

    gt_df = collect_gff(str(gt_path), exclude_features=excl_list)
    pred_dfs = {name: collect_gff(str(p), exclude_features=excl_list) for name, p in pred_paths.items()}

    # ------------------------------------------------------------------
    # 3. Map GT <-> predictions
    # ------------------------------------------------------------------
    click.echo("Mapping transcripts (locus-based junction-F1 matching)...")

    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        label_config=label_config,
        transcript_types=tt_list,
        exclude_features=excl_list,
        locus_matching_mode=mode,
        **gt_map_kwargs,
        **pred_map_kwargs,
    )

    if not mappings:
        raise click.ClickException(
            "No transcript mappings found.  Check that the GT and "
            "prediction files share overlapping genomic regions on "
            "the same strand."
        )

    if mapping_output_path is not None:
        export_mapping_table(mappings, mapping_output_path)
        click.echo(f"Mapping table written to {mapping_output_path}")

    # ------------------------------------------------------------------
    # 4. Build arrays and benchmark each predictor
    # ------------------------------------------------------------------
    gt_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}
    pred_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}

    for mapping in mappings:
        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=label_config,
            transcript_types=tt_list,
            **gt_map_kwargs,
            **pred_map_kwargs,
        )

        for pred_name in pred_paths:
            has_match = any(m.predictor_name == pred_name for m in mapping.matched_predictions)
            if mapping.is_unmatched_prediction:
                if not has_match:
                    continue
            elif mode == LocusMatchingMode.BEST_PER_LOCUS and not has_match:
                continue

            gt_by_pred[pred_name].append(gt_arr)
            pred_by_pred[pred_name].append(pred_arrs[pred_name])

    all_results: dict[str, dict] = {}

    for pred_name in pred_paths:
        gt_labels = gt_by_pred[pred_name]
        pred_labels = pred_by_pred[pred_name]

        if not gt_labels:
            click.echo(f"  Warning: No mapped transcripts for predictor '{pred_name}', skipping.")
            continue

        eval_classes = list(label_config.evaluation_labels.keys())
        if use_role_maps:
            gt_feature_desc = gt_role_map if gt_role_map is not None else "mode default"
            pred_feature_desc = _pred_role_map_for(pred_role_maps, pred_name)
            pred_feature_desc = pred_feature_desc if pred_feature_desc is not None else "GT map"
        else:
            gt_feature_desc = gt_exon_types
            pred_feature_desc = pred_exon_types_by_name[pred_name]
        click.echo(
            f"  Benchmarking '{pred_name}': "
            f"{len(gt_labels)} transcript(s) | "
            f"mode={label_config.annotation_mode.value} | "
            f"classes={[label_config.name_of(c) for c in eval_classes]} | "
            f"metrics={[m.name for m in metrics]} | "
            f"gt_features={gt_feature_desc} | "
            f"pred_features={pred_feature_desc}"
        )

        aggregated = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            label_config=label_config,
            metrics=metrics,
            return_individual_results=individual,
            infer_introns=infer_introns,
        )

        # Individual mode returns a list, not a dict — skip global aggregation.
        if individual:
            all_results[pred_name] = aggregated
            continue

        _pred_role_map = _pred_role_map_for(pred_role_maps, pred_name)
        global_result = compute_global_metrics(
            gt_df=gt_df,
            pred_df=pred_dfs[pred_name],
            mappings=mappings,
            predictor_name=pred_name,
            label_config=label_config,
            transcript_types=tt_list,
            gt_feature_role_map=gt_role_map,
            pred_feature_role_map=_pred_role_map if _pred_role_map is not None else gt_role_map,
        )

        all_results[pred_name] = {
            "aggregated": aggregated,
            "global": global_result,
        }

    # ------------------------------------------------------------------
    # 5. Serialise and output
    # ------------------------------------------------------------------
    serialised = _serialise_results(all_results)
    output_json = json.dumps(serialised, indent=2)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json)
        click.echo(f"Results written to {output_path}")
    else:
        click.echo(output_json)


# ------------------------------------------------------------------
# `init-config` command
# ------------------------------------------------------------------


_EXON_INTRON_TEMPLATE = {
    "annotation_mode": "EXON_INTRON",
    "background_label": 8,
    "exon_label": 0,
    "intron_label": 2,
    "splice_donor_label": 1,
    "splice_acceptor_label": 3,
}

_UTR_CDS_INTRON_TEMPLATE = {
    "annotation_mode": "UTR_CDS_INTRON",
    "background_label": 8,
    "cds_label": 0,
    "five_prime_utr_label": 4,
    "three_prime_utr_label": 5,
    "intron_label": 2,
    "splice_donor_label": 1,
    "splice_acceptor_label": 3,
}

_CONFIG_TEMPLATES = {
    "exon_intron": _EXON_INTRON_TEMPLATE,
    "utr_cds_intron": _UTR_CDS_INTRON_TEMPLATE,
}

_TEMPLATE_HINTS = {
    "exon_intron": (
        "# One exonic class; optional intron/splice labels.\n"
        "# Map GFF/GTF feature names with CLI flags:\n"
        "#   --gt-exon-feature-type exon\n"
        "#   --pred-exon-feature-type augustus:CDS --pred-exon-feature-type helixer:exon\n"
    ),
    "utr_cds_intron": (
        "# Distinct 5' UTR / CDS / 3' UTR classes; enables CDS-scoped metrics\n"
        "# and PHASE_DRIFT. Map GFF/GTF feature names to roles with CLI flags:\n"
        "#   --gt-feature-role five_prime_UTR:five_prime_utr\n"
        "#   --gt-feature-role CDS:cds\n"
        "#   --gt-feature-role three_prime_UTR:three_prime_utr\n"
        "# Per-predictor maps use predictor=feature_type:role, e.g.\n"
        "#   --pred-feature-role helixer=CDS:cds\n"
    ),
}


@cli.command("init-config")
@click.option(
    "--mode",
    "mode",
    type=click.Choice(list(_CONFIG_TEMPLATES.keys()), case_sensitive=False),
    default="exon_intron",
    show_default=True,
    help=(
        "Annotation mode for the template.  'exon_intron' uses a single exonic "
        "label; 'utr_cds_intron' uses distinct 5' UTR / CDS / 3' UTR labels."
    ),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("label_config.yaml"),
    show_default=True,
    help="Where to write the template config.",
)
def init_config(mode: str, output_path: Path):
    """Generate a template label_config.yaml file for the chosen mode."""
    import yaml

    template = _CONFIG_TEMPLATES[mode.lower()]
    content = yaml.dump(template, sort_keys=False)
    content += _TEMPLATE_HINTS[mode.lower()]
    output_path.write_text(content)
    click.echo(f"Template config ({template['annotation_mode']}) written to {output_path}")
    click.echo("Edit the file to match your label set, then use: dna-benchmark run --config " + str(output_path))
