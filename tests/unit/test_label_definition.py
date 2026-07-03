import numpy as np
import pytest
from pydantic import ValidationError

from dna_segmentation_benchmark.eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_single,
)
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    BEND_LABEL_CONFIG,
    LabelConfig,
)


class TestLabelConfigConstruction:
    def test_exon_intron_minimal_config(self):
        config = LabelConfig(
            annotation_mode=AnnotationMode.EXON_INTRON,
            background_label=8,
            exon_label=0,
        )

        assert config.annotation_mode is AnnotationMode.EXON_INTRON
        assert config.evaluation_scope is BenchmarkScope.TRANSCRIPT_EXON
        assert config.available_scopes() == (BenchmarkScope.TRANSCRIPT_EXON,)
        assert config.scope_tokens(BenchmarkScope.TRANSCRIPT_EXON) == frozenset({0})
        assert config.labels == {8: "NONCODING", 0: "EXON"}
        assert config.evaluation_labels == {0: "EXON"}
        assert config.supports_phase_drift is False

    def test_utr_cds_minimal_config(self):
        config = LabelConfig(
            annotation_mode=AnnotationMode.UTR_CDS_INTRON,
            background_label=8,
            cds_label=0,
            five_prime_utr_label=4,
            three_prime_utr_label=5,
        )

        assert config.annotation_mode is AnnotationMode.UTR_CDS_INTRON
        assert config.evaluation_scope is BenchmarkScope.TRANSCRIPT_EXON
        assert config.available_scopes() == (
            BenchmarkScope.TRANSCRIPT_EXON,
            BenchmarkScope.CDS,
        )
        assert config.scope_tokens(BenchmarkScope.TRANSCRIPT_EXON) == frozenset({4, 0, 5})
        assert config.scope_tokens(BenchmarkScope.CDS) == frozenset({0})
        assert config.labels == {
            8: "NONCODING",
            4: "FIVE_PRIME_UTR",
            0: "CDS",
            5: "THREE_PRIME_UTR",
        }
        assert config.evaluation_labels == {
            4: "FIVE_PRIME_UTR",
            0: "CDS",
            5: "THREE_PRIME_UTR",
        }
        assert config.supports_phase_drift is False

    def test_optional_labels_are_included_in_labels_and_evaluation_labels(self):
        config = LabelConfig(
            annotation_mode=AnnotationMode.UTR_CDS_INTRON,
            evaluation_scope=BenchmarkScope.CDS,
            background_label=8,
            cds_label=0,
            five_prime_utr_label=4,
            three_prime_utr_label=5,
            intron_label=2,
            splice_donor_label=1,
            splice_acceptor_label=3,
        )

        assert config.labels == {
            8: "NONCODING",
            4: "FIVE_PRIME_UTR",
            0: "CDS",
            5: "THREE_PRIME_UTR",
            2: "INTRON",
            1: "SPLICE_DONOR",
            3: "SPLICE_ACCEPTOR",
        }
        assert config.evaluation_labels == {
            4: "FIVE_PRIME_UTR",
            0: "CDS",
            5: "THREE_PRIME_UTR",
            2: "INTRON",
            1: "SPLICE_DONOR",
            3: "SPLICE_ACCEPTOR",
        }
        assert config.supports_phase_drift is True

    def test_name_of_returns_label_name_and_falls_back_to_token_string(self):
        assert BEND_LABEL_CONFIG.name_of(0) == "EXON"
        assert BEND_LABEL_CONFIG.name_of(2) == "INTRON"
        assert BEND_LABEL_CONFIG.name_of(99) == "99"

    def test_config_is_frozen(self):
        with pytest.raises(ValidationError, match="frozen"):
            BEND_LABEL_CONFIG.background_label = 99


class TestLabelConfigValidation:
    def test_exon_intron_requires_exon_label(self):
        with pytest.raises(ValueError, match="requires exon_label"):
            LabelConfig(
                annotation_mode=AnnotationMode.EXON_INTRON,
                background_label=8,
            )

    @pytest.mark.parametrize(
        "field_name, field_value",
        [
            ("cds_label", 0),
            ("five_prime_utr_label", 4),
            ("three_prime_utr_label", 5),
        ],
    )
    def test_exon_intron_rejects_utr_cds_fields(self, field_name, field_value):
        with pytest.raises(ValueError, match="does not allow"):
            LabelConfig(
                annotation_mode=AnnotationMode.EXON_INTRON,
                background_label=8,
                exon_label=1,
                **{field_name: field_value},
            )

    @pytest.mark.parametrize(
        "missing_field, kwargs",
        [
            ("cds_label", {"five_prime_utr_label": 4, "three_prime_utr_label": 5}),
            ("five_prime_utr_label", {"cds_label": 0, "three_prime_utr_label": 5}),
            ("three_prime_utr_label", {"cds_label": 0, "five_prime_utr_label": 4}),
        ],
    )
    def test_utr_cds_requires_all_explicit_roles(self, missing_field, kwargs):
        with pytest.raises(ValueError, match=missing_field):
            LabelConfig(
                annotation_mode=AnnotationMode.UTR_CDS_INTRON,
                background_label=8,
                **kwargs,
            )

    def test_utr_cds_rejects_exon_label(self):
        with pytest.raises(ValueError, match="does not allow exon_label"):
            LabelConfig(
                annotation_mode=AnnotationMode.UTR_CDS_INTRON,
                background_label=8,
                exon_label=7,
                cds_label=0,
                five_prime_utr_label=4,
                three_prime_utr_label=5,
            )

    def test_exon_intron_rejects_cds_evaluation_scope(self):
        with pytest.raises(ValueError, match="evaluation_scope"):
            LabelConfig(
                annotation_mode=AnnotationMode.EXON_INTRON,
                evaluation_scope=BenchmarkScope.CDS,
                background_label=8,
                exon_label=0,
            )

    def test_splice_site_labels_must_appear_as_pair(self):
        with pytest.raises(ValueError, match="must either both be set or both be omitted"):
            LabelConfig(
                annotation_mode=AnnotationMode.EXON_INTRON,
                background_label=8,
                exon_label=0,
                splice_donor_label=1,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"background_label": 0, "exon_label": 0},
            {"background_label": 8, "exon_label": 0, "intron_label": 0},
            {
                "background_label": 8,
                "cds_label": 0,
                "five_prime_utr_label": 4,
                "three_prime_utr_label": 4,
            },
        ],
    )
    def test_duplicate_labels_raise(self, kwargs):
        annotation_mode = (
            AnnotationMode.EXON_INTRON
            if "exon_label" in kwargs
            else AnnotationMode.UTR_CDS_INTRON
        )
        with pytest.raises(ValueError, match="duplicates"):
            LabelConfig(annotation_mode=annotation_mode, **kwargs)


class TestLabelConfigHelpers:
    @pytest.mark.parametrize(
        "scope",
        [
            BenchmarkScope.TRANSCRIPT_EXON,
            "transcript_exon",
        ],
    )
    def test_scope_tokens_accept_enum_and_string_for_transcript_scope(self, scope):
        assert BEND_LABEL_CONFIG.scope_tokens(scope) == frozenset({0})

    @pytest.mark.parametrize(
        "scope",
        [
            BenchmarkScope.CDS,
            "cds",
        ],
    )
    def test_scope_tokens_accept_enum_and_string_for_cds_scope(self, scope):
        config = LabelConfig(
            annotation_mode=AnnotationMode.UTR_CDS_INTRON,
            background_label=8,
            cds_label=0,
            five_prime_utr_label=4,
            three_prime_utr_label=5,
        )
        assert config.scope_tokens(scope) == frozenset({0})

    def test_scope_tokens_reject_cds_scope_for_exon_intron_mode(self):
        with pytest.raises(ValueError, match="not available"):
            BEND_LABEL_CONFIG.scope_tokens(BenchmarkScope.CDS)

    def test_scope_tokens_reject_unknown_scope(self):
        with pytest.raises(ValueError, match="Unknown benchmark scope"):
            BEND_LABEL_CONFIG.scope_tokens("bogus")

    def test_phase_drift_requires_utr_cds_cds_scope(self):
        with pytest.raises(ValueError, match="UTR_CDS_INTRON"):
            benchmark_gt_vs_pred_single(
                gt_labels=np.array([8, 0, 0, 8]),
                pred_labels=np.array([8, 0, 0, 8]),
                label_config=BEND_LABEL_CONFIG,
                metrics=[EvalMetrics.PHASE_DRIFT],
            )


class TestDefaultFactories:
    def test_default_exon_intron_matches_bend(self):
        """The EXON_INTRON factory reproduces the canonical BEND scheme."""
        config = LabelConfig.default_exon_intron()
        assert config == BEND_LABEL_CONFIG
        assert config.available_scopes() == (BenchmarkScope.TRANSCRIPT_EXON,)

    def test_default_utr_cds_intron_tokens_and_scopes(self):
        config = LabelConfig.default_utr_cds_intron()
        assert config.annotation_mode is AnnotationMode.UTR_CDS_INTRON
        assert config.cds_label == 0
        assert config.five_prime_utr_label == 4
        assert config.three_prime_utr_label == 5
        # CDS scope is available (precondition for PHASE_DRIFT once selected).
        assert BenchmarkScope.CDS in config.available_scopes()
        # All configured tokens are distinct (validators ran without error).
        tokens = [t for t, _ in config._configured_labels()]
        assert len(tokens) == len(set(tokens))
