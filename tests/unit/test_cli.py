"""CLI surface tests: feature-spec parsers, ``init-config``, ``datasets``.

``test_cli_parity.py`` already covers the ``run`` command end-to-end; this file
pins the parts of ``cli.py`` that ``run`` only touches indirectly — the
feature-role / exon-type spec parsers (including their ``BadParameter``
branches), the ``init-config`` generator, and the local ``datasets`` commands
(``list`` / ``info``, no network).
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from dna_segmentation_benchmark.cli import (
    _load_label_config,
    _parse_gt_feature_role_specs,
    _parse_pred_exon_feature_specs,
    _parse_pred_feature_role_specs,
    _pred_role_map_for,
    _resolve_pred_exon_types,
    cli,
)


# ---------------------------------------------------------------------------
# --pred-exon-feature-type
# ---------------------------------------------------------------------------


def test_pred_exon_specs_empty_is_none():
    assert _parse_pred_exon_feature_specs(()) is None


def test_pred_exon_specs_plain_applies_to_all():
    assert _parse_pred_exon_feature_specs(("CDS",)) == ["CDS"]
    assert _parse_pred_exon_feature_specs(("exon", "CDS")) == ["exon", "CDS"]


def test_pred_exon_specs_named_builds_per_predictor():
    parsed = _parse_pred_exon_feature_specs(("augustus:CDS", "helixer:exon"))
    assert parsed == {"augustus": "CDS", "helixer": "exon"}


def test_pred_exon_specs_mixed_plain_and_named_raises():
    with pytest.raises(click.BadParameter):
        _parse_pred_exon_feature_specs(("CDS", "helixer:exon"))


def test_pred_exon_specs_malformed_named_raises():
    with pytest.raises(click.BadParameter):
        _parse_pred_exon_feature_specs(("augustus:",))


# ---------------------------------------------------------------------------
# --gt-feature-role / --pred-feature-role
# ---------------------------------------------------------------------------


def test_gt_feature_role_specs_valid_and_empty():
    assert _parse_gt_feature_role_specs(()) is None
    assert _parse_gt_feature_role_specs(("CDS:cds", "five_prime_UTR:five_prime_utr")) == {
        "CDS": "cds",
        "five_prime_UTR": "five_prime_utr",
    }


def test_gt_feature_role_specs_malformed_raises():
    with pytest.raises(click.BadParameter):
        _parse_gt_feature_role_specs(("CDS",))  # no :role


def test_pred_feature_role_specs_plain():
    assert _parse_pred_feature_role_specs(("CDS:cds",)) == {"CDS": "cds"}


def test_pred_feature_role_specs_named_per_predictor():
    parsed = _parse_pred_feature_role_specs(("helixer=CDS:cds", "augustus=exon:cds"))
    assert parsed == {"helixer": {"CDS": "cds"}, "augustus": {"exon": "cds"}}


def test_pred_feature_role_specs_mixed_raises():
    with pytest.raises(click.BadParameter):
        _parse_pred_feature_role_specs(("CDS:cds", "helixer=exon:cds"))


def test_pred_feature_role_specs_malformed_named_raises():
    with pytest.raises(click.BadParameter):
        _parse_pred_feature_role_specs(("helixer=CDS",))  # pair lacks :role


# ---------------------------------------------------------------------------
# _resolve_pred_exon_types / _pred_role_map_for
# ---------------------------------------------------------------------------


def test_resolve_pred_exon_types_default_for_all():
    assert _resolve_pred_exon_types(None, ["a", "b"], ["exon"]) == {"a": ["exon"], "b": ["exon"]}


def test_resolve_pred_exon_types_dict_falls_back_to_default():
    resolved = _resolve_pred_exon_types({"a": "CDS"}, ["a", "b"], ["exon"])
    assert resolved == {"a": ["CDS"], "b": ["exon"]}  # b unnamed → default


def test_resolve_pred_exon_types_plain_list_applies_to_all():
    assert _resolve_pred_exon_types(["CDS"], ["a", "b"], ["exon"]) == {"a": ["CDS"], "b": ["CDS"]}


def test_pred_role_map_for_plain_applies_to_every_predictor():
    plain = {"CDS": "cds"}
    assert _pred_role_map_for(plain, "anything") == plain


def test_pred_role_map_for_nested_indexes_by_name():
    nested = {"helixer": {"CDS": "cds"}}
    assert _pred_role_map_for(nested, "helixer") == {"CDS": "cds"}
    assert _pred_role_map_for(nested, "augustus") is None


def test_pred_role_map_for_none():
    assert _pred_role_map_for(None, "x") is None


# ---------------------------------------------------------------------------
# init-config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,expected", [("exon_intron", "EXON_INTRON"), ("utr_cds_intron", "UTR_CDS_INTRON")])
def test_init_config_writes_loadable_template(mode, expected, tmp_path):
    out = tmp_path / "cfg.yaml"
    result = CliRunner().invoke(cli, ["init-config", "--mode", mode, "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    # Round-trip through the loader the `run` command uses — proves the emitted
    # template is a valid LabelConfig, not just well-formed YAML.
    config = _load_label_config(out)
    assert config.annotation_mode.name == expected


# ---------------------------------------------------------------------------
# datasets (local registry, no network)
# ---------------------------------------------------------------------------


def test_datasets_list_shows_registered_entry():
    result = CliRunner().invoke(cli, ["datasets", "list"])
    assert result.exit_code == 0, result.output
    assert "zenodo_test" in result.output


def test_datasets_info_known_entry():
    result = CliRunner().invoke(cli, ["datasets", "info", "zenodo_test"])
    assert result.exit_code == 0, result.output
    assert "zenodo_test" in result.output
    assert "record:" in result.output


def test_datasets_info_unknown_entry_errors():
    result = CliRunner().invoke(cli, ["datasets", "info", "does_not_exist"])
    assert result.exit_code != 0
