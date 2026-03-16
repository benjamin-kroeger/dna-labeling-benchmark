"""Command-line interface for the DNA segmentation benchmark.

Usage examples::

    # Basic run with a single prediction
    dna-benchmark run \\
        --gt ground_truth.gff \\
        --pred augustus:predictions.gff \\
        --config label_config.yaml \\
        --classes 0 2 \\
        --metrics REGION_DISCOVERY NUCLEOTIDE_CLASSIFICATION

    # Compare multiple predictors at once
    dna-benchmark run \\
        --gt ground_truth.gff \\
        --pred augustus:augustus.gff \\
        --pred helixer:helixer.gff \\
        --config label_config.yaml \\
        --classes 0 \\
        --overlap-threshold 0.2 \\
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
from .label_definition import LabelConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_METRIC_NAMES = {m.name: m for m in EvalMetrics if not m.name.startswith("_")}


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

    Expected YAML structure::

        labels:
          0: EXON
          2: INTRON
          8: NONCODING
        background_label: 8
        coding_label: 0
        splice_donor_label: null
        splice_acceptor_label: null
    """
    import yaml
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    # Convert the `labels` keys to ints (YAML may parse them as strings)
    if "labels" in raw and isinstance(raw["labels"], dict):
        raw["labels"] = {int(k): v for k, v in raw["labels"].items()}

    return LabelConfig(**raw)


def _load_npz_to_list(path: Path) -> list[np.ndarray]:
    """Load an ``.npz`` file and return its arrays as a list."""
    npz = np.load(str(path), allow_pickle=True)
    return [npz[key] for key in npz.keys()]


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
    "--gt", "gt_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to ground-truth annotations (.gff3 or .gtf file).",
)
@click.option(
    "--pred", "pred_specs",
    required=True,
    type=str,
    multiple=True,
    help=(
        "Prediction file(s).  Format: 'name:path' or just 'path' "
        "(repeatable, e.g. --pred augustus:aug.gff --pred helixer:hel.gff)."
    ),
)
@click.option(
    "--config", "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to label configuration file (.yaml).",
)
@click.option(
    "--exclude-features", "exclude_features",
    type=str,
    multiple=True,
    help="GFF feature types to ignore (e.g. --exclude-features gene).",
)
@click.option(
    "--transcript-types", "transcript_types",
    type=str,
    multiple=True,
    default=["mRNA", "transcript"],
    show_default=True,
    help="Feature types that define transcript boundaries (repeatable).",
)
@click.option(
    "--classes", "class_tokens",
    required=True,
    type=int,
    multiple=True,
    help=(
        "Integer token(s) to evaluate "
        "(repeatable, e.g. --classes 0 --classes 2)."
    ),
)
@click.option(
    "--metrics", "metric_names",
    type=click.Choice(list(_METRIC_NAMES.keys()), case_sensitive=False),
    multiple=True,
    default=["REGION_DISCOVERY", "NUCLEOTIDE_CLASSIFICATION"],
    show_default=True,
    help="Metric group(s) to compute (repeatable).",
)
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write results to this JSON file instead of stdout.",
)
@click.option(
    "--individual", is_flag=True, default=False,
    help="Return per-sequence results instead of aggregated.",
)
@click.option(
    "--overlap-threshold", "overlap_threshold",
    type=float,
    default=0.2,
    show_default=True,
    help="Minimum overlap fraction (0-1) to map a prediction to a GT transcript.",
)
@click.option(
    "--mapping-output", "mapping_output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the GT <-> prediction mapping to this TSV file for debugging.",
)
def run(
    gt_path: Path,
    pred_specs: tuple[str, ...],
    config_path: Path,
    exclude_features: tuple[str, ...],
    transcript_types: tuple[str, ...],
    class_tokens: tuple[int, ...],
    metric_names: tuple[str, ...],
    output_path: Path | None,
    individual: bool,
    overlap_threshold: float,
    mapping_output_path: Path | None,
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

    for token in class_tokens:
        if token not in label_config.labels:
            raise click.BadParameter(
                f"Token {token} is not defined in the label config. "
                f"Available: {list(label_config.labels.keys())}",
                param_hint="--classes",
            )

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
                f"Duplicate predictor name '{name}'. Use 'name:path' "
                f"format to give each prediction a unique name.",
                param_hint="--pred",
            )
        pred_paths[name] = path

    excl_list = list(exclude_features)
    tt_list = list(transcript_types)
    metrics = [_METRIC_NAMES[name.upper()] for name in metric_names]

    # ------------------------------------------------------------------
    # 2. Read GFF files once into memory
    # ------------------------------------------------------------------
    click.echo(
        f"Reading GT: {gt_path.name} and "
        f"{len(pred_paths)} prediction file(s)..."
    )

    gt_df = collect_gff(str(gt_path), exclude_features=excl_list)
    pred_dfs = {
        name: collect_gff(str(p), exclude_features=excl_list)
        for name, p in pred_paths.items()
    }

    # ------------------------------------------------------------------
    # 3. Map GT <-> predictions
    # ------------------------------------------------------------------
    click.echo(
        f"Mapping transcripts (overlap >= {overlap_threshold:.0%})..."
    )

    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        min_overlap=overlap_threshold,
        transcript_types=tt_list,
        exclude_features=excl_list,
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
    bg_val = label_config.background_label

    # Collect arrays per-predictor across all mappings
    gt_by_pred: dict[str, list[np.ndarray]] = {
        name: [] for name in pred_paths
    }
    pred_by_pred: dict[str, list[np.ndarray]] = {
        name: [] for name in pred_paths
    }

    for mapping in mappings:
        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=label_config,
            transcript_types=tt_list,
        )

        for pred_name in pred_paths:
            # For unmatched-prediction mappings, only include the
            # predictor that owns the unmatched prediction.
            if mapping.is_unmatched_prediction:
                owns_mapping = any(
                    m.predictor_name == pred_name
                    for m in mapping.matched_predictions
                )
                if not owns_mapping:
                    continue

            gt_by_pred[pred_name].append(gt_arr)
            pred_by_pred[pred_name].append(pred_arrs[pred_name])

    all_results: dict[str, dict] = {}

    for pred_name in pred_paths:
        gt_labels = gt_by_pred[pred_name]
        pred_labels = pred_by_pred[pred_name]

        if not gt_labels:
            click.echo(
                f"  Warning: No mapped transcripts for "
                f"predictor '{pred_name}', skipping."
            )
            continue

        click.echo(
            f"  Benchmarking '{pred_name}': "
            f"{len(gt_labels)} transcript(s) | "
            f"classes={list(class_tokens)} | "
            f"metrics={[m.name for m in metrics]}"
        )

        results = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            label_config=label_config,
            classes=list(class_tokens),
            metrics=metrics,
            return_individual_results=individual,
        )

        all_results[pred_name] = results

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

@cli.command("init-config")
@click.option(
    "--output", "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("label_config.yaml"),
    show_default=True,
    help="Where to write the template config.",
)
def init_config(output_path: Path):
    """Generate a template label_config.yaml file."""
    template = {
        "labels": {
            "0": "EXON",
            "1": "DONOR",
            "2": "INTRON",
            "3": "ACCEPTOR",
            "8": "NONCODING",
        },
        "background_label": 8,
        "coding_label": 0,
        "splice_donor_label": 1,
        "splice_acceptor_label": 3,
    }
    import yaml
    output_path.write_text(yaml.dump(template, sort_keys=False) + "\n")
    click.echo(f"Template config written to {output_path}")
    click.echo(
        "Edit the file to match your label set, then use: "
        "dna-benchmark run --config " + str(output_path)
    )