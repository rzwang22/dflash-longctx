from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_phase2d2a_h2_freeze import check_h2_freeze
from scripts.freeze_phase2d2a_h2_cohort import (
    CANDIDATE_LABEL_AUTHORITY,
    CANDIDATE_LABEL_SOURCE,
    EXECUTION_IDENTITY_ARTIFACT_NAMES,
    FAMILIES,
    FROZEN_REFINEMENT_FAMILIES,
    INDEPENDENT_TASKS,
    INVENTORY_LENGTHS,
    LONG_CONTEXT_ORDER,
    SELECTION_SEED,
    TASK_ROLES,
    TASK_SAMPLE_COUNTS,
    TASKS,
    VERIFICATION_MODE,
    EligibilityError,
    build_aggregation_rules,
    build_freeze_artifacts,
    build_matrix,
    build_task_inventory,
    choose_common_long_context,
    collect_exclusion_provenance,
    deterministic_select,
    exclusion_index,
    exclusion_matches,
    freeze_cohort,
    identity_from_execution_record,
    persist_freeze,
    persist_ineligible_inventory,
    physical_selection_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return [ord(character) for character in messages[0]["content"]]


def _candidate(task: str, index: int, length: int) -> dict:
    source_id = f"{task}-source-{index}"
    source_hash = hashlib.sha256(f"{task}-text-{index}".encode()).hexdigest()
    return {
        "sample_id": f"{task}-sample-{index}",
        "source_sample_id": source_id,
        "source_text_sha256": source_hash,
        "source_index": index,
        "document_token_length": 20000 + index,
        "source_full_prompt_length": 20000 + index,
        "fixed_prompt_length": 32,
        "selection_hash": physical_selection_hash(
            task=task,
            source_sample_id=source_id,
            source_text_sha256=source_hash,
        ),
        "actual_prompt_length": length,
        "nominal_prompt_length_shortfall": 0,
        "context_prefix_characters": length - 32,
        "context_prefix_sha256": hashlib.sha256(
            f"{task}-{index}-{length}-prefix".encode()
        ).hexdigest(),
        "input_token_ids_sha256": hashlib.sha256(
            f"{task}-{index}-{length}-tokens".encode()
        ).hexdigest(),
    }


def _task_inventory(task: str, counts: dict[int, int]) -> dict:
    contexts = []
    for length in INVENTORY_LENGTHS:
        eligible = [_candidate(task, index, length) for index in range(counts[length])]
        eligible.sort(key=lambda row: (row["selection_hash"], row["sample_id"]))
        contexts.append(
            {
                "nominal_prompt_length": length,
                "eligible_before_exclusion_count": len(eligible),
                "excluded_previous_sample_count": 0,
                "eligible_sample_count": len(eligible),
                "eligible_document_token_length_distribution": {
                    "count": len(eligible),
                    "min": min(
                        (row["document_token_length"] for row in eligible),
                        default=None,
                    ),
                    "median": None,
                    "p90": None,
                    "max": max(
                        (row["document_token_length"] for row in eligible),
                        default=None,
                    ),
                },
                "eligible_sample_ids": [row["sample_id"] for row in eligible],
                "eligible_source_sample_ids": [
                    row["source_sample_id"] for row in eligible
                ],
                "eligible_source_text_sha256": [
                    row["source_text_sha256"] for row in eligible
                ],
                "excluded_samples": [],
                "eligible_samples": eligible,
            }
        )
    return {
        "task": task,
        "task_role": TASK_ROLES[task],
        "requested_physical_sample_count": TASK_SAMPLE_COUNTS[task],
        "source_record_count": counts[1024],
        "usable_record_count": counts[1024],
        "unusable_record_count": 0,
        "source_document_token_length_distribution": {
            "count": counts[1024],
            "min": 20000,
            "median": 20001.0,
            "p90": 20002.0,
            "max": 20020,
        },
        "prompt_template": {"name": task, "version": "1", "sha256": "a" * 64},
        "contexts": contexts,
    }


def _inventory(
    *,
    qasper_16k: int = 4,
    hotpot_16k: int = 9,
    qasper_8k: int = 5,
    hotpot_8k: int = 10,
) -> dict:
    counts = {
        "gov_report": {1024: 5, 4096: 5, 8192: 5, 16384: 4, 32768: 0},
        "qasper": {
            1024: 6,
            4096: 6,
            8192: qasper_8k,
            16384: qasper_16k,
            32768: 0,
        },
        "hotpotqa": {
            1024: 12,
            4096: 11,
            8192: hotpot_8k,
            16384: hotpot_16k,
            32768: 0,
        },
    }
    return {
        "schema_version": "1.0",
        "phase": "2D.2A-H2.0",
        "status": "complete",
        "mode": "target_blind_token_length_inventory",
        "longbench_root": "/datasets/LongBench/data",
        "dataset_split": "test",
        "inventory_context_lengths": list(INVENTORY_LENGTHS),
        "primary_cross_task_32k_anchor_allowed": False,
        "model_forward_executed": False,
        "cuda_required": False,
        "exclusion_applied_before_selection": True,
        "selection_seed": SELECTION_SEED,
        "tasks": [_task_inventory(task, counts[task]) for task in TASKS],
    }


def _exclusions(*identities: dict) -> dict:
    if not identities:
        identities = (
            {
                "task": "gov_report",
                "sample_id": "gov_report-fd287bbfd3bdbdc9",
                "source_sample_id": "previous-h1-source",
                "source_text_sha256": "f" * 64,
                "evidence": [
                    {"artifact_path": "/results/samples.jsonl", "line_number": 1}
                ],
            },
        )
    return {
        "schema_version": "1.0",
        "phase": "2D.2A-H2.0",
        "status": "complete",
        "target_blind": True,
        "selection_or_exclusion_uses_model_outputs": False,
        "identity_fields_read": [
            "task",
            "longbench_task",
            "sample_id",
            "source_sample_id",
            "source_text_sha256",
        ],
        "forbidden_outcome_fields_read": False,
        "forbidden_outcome_fields": [
            "generated_output_token_ids",
            "accepted_draft_token_count",
            "candidate_quality",
            "latency",
            "target_labels",
        ],
        "execution_identity_artifact_names": sorted(EXECUTION_IDENTITY_ARTIFACT_NAMES),
        "result_roots": ["/results"],
        "scanned_artifacts": [
            {
                "artifact_path": "/results/samples.jsonl",
                "artifact_sha256": "a" * 64,
                "record_count": 1,
                "identity_record_count": 1,
            }
        ],
        "scanned_artifact_count": 1,
        "excluded_identity_count": len(identities),
        "excluded_identities": list(identities),
        "exclusion_completed_before_h2_model_output_exists": True,
    }


def _artifacts() -> dict:
    return build_freeze_artifacts(
        inventory=_inventory(),
        exclusions=_exclusions(),
        tokenizer_provenance={
            "model_id": "Qwen/Qwen3-8B",
            "revision": "b" * 40,
            "model_loaded": False,
        },
        h11_provenance={
            "checker_status": "accepted",
            "h2_execution_authorized_by_h11": False,
        },
    )


class Phase2D2H2FreezeTests(unittest.TestCase):
    def test_previous_used_samples_are_excluded_by_identity(self) -> None:
        identity = {
            "task": "qasper",
            "sample_id": "qasper-used",
            "source_sample_id": "source-used",
            "source_text_sha256": "a" * 64,
            "evidence": [],
        }
        index = exclusion_index(_exclusions(identity))
        matches = exclusion_matches(
            {
                "longbench_task": "qasper",
                "sample_id": "qasper-used",
                "source_sample_id": "source-used",
                "source_text_sha256": "a" * 64,
            },
            index,
        )
        self.assertEqual(
            matches, ["sample_id", "source_sample_id", "source_text_sha256"]
        )

    def test_exclusion_collection_uses_identity_not_model_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {
                    "task": "qasper",
                    "sample_id": "qasper-used",
                    "source_sample_id": "source-used",
                    "source_text_sha256": "a" * 64,
                    "accepted_draft_token_count": value,
                    "latency": 1000 - value,
                    "target_labels": [value],
                }
                for value in (1, 99)
            ]
            (root / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            provenance = collect_exclusion_provenance([root])
            self.assertEqual(provenance["excluded_identity_count"], 1)
            self.assertEqual(len(provenance["excluded_identities"][0]["evidence"]), 2)
            self.assertFalse(provenance["forbidden_outcome_fields_read"])

    def test_identity_parser_ignores_outcome_values(self) -> None:
        base = {
            "task": "hotpotqa",
            "sample_id": "hotpotqa-1",
            "source_sample_id": "source-1",
            "source_text_sha256": "b" * 64,
        }
        first = identity_from_execution_record({**base, "output": [1], "latency": 2})
        second = identity_from_execution_record(
            {**base, "output": [999], "latency": 99999}
        )
        self.assertEqual(first, second)

    def test_common_long_context_order_is_16k_then_8k_then_4k(self) -> None:
        self.assertEqual(LONG_CONTEXT_ORDER, (16384, 8192, 4096))
        selected, audit = choose_common_long_context(
            _inventory(qasper_16k=2, hotpot_16k=7)
        )
        self.assertEqual(selected, 8192)
        self.assertEqual([row["nominal_prompt_length"] for row in audit], [16384, 8192])

    def test_common_context_choice_is_independent_of_model_outputs(self) -> None:
        inventory = _inventory()
        expected = choose_common_long_context(inventory)[0]
        for task in inventory["tasks"]:
            for context in task["contexts"]:
                for sample in context["eligible_samples"]:
                    sample["accepted_tokens"] = -1000
                    sample["target_output"] = "adversarial"
        self.assertEqual(choose_common_long_context(inventory)[0], expected)

    def test_requested_task_counts_are_frozen(self) -> None:
        cohort = freeze_cohort(_inventory())
        counts = {
            row["task"]: row["requested_physical_sample_count"]
            for row in cohort["tasks"]
        }
        self.assertEqual(counts, {"gov_report": 3, "qasper": 3, "hotpotqa": 8})

    def test_insufficient_common_context_stops_without_reduction(self) -> None:
        inventory = _inventory(qasper_16k=2, hotpot_16k=7, qasper_8k=2, hotpot_8k=7)
        for task in inventory["tasks"]:
            if task["task"] in INDEPENDENT_TASKS:
                context = next(
                    row
                    for row in task["contexts"]
                    if row["nominal_prompt_length"] == 4096
                )
                context["eligible_samples"] = context["eligible_samples"][:2]
                context["eligible_sample_count"] = 2
        with self.assertRaisesRegex(EligibilityError, "none of 16K, 8K, or 4K"):
            choose_common_long_context(inventory)

    def test_deterministic_hash_selection_is_stable(self) -> None:
        candidates = [_candidate("qasper", index, 16384) for index in range(8)]
        first = deterministic_select(candidates, count=3)
        second = deterministic_select(list(reversed(candidates)), count=3)
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertTrue(
            all(
                row["selection_hash"]
                == physical_selection_hash(
                    task="qasper",
                    source_sample_id=row["source_sample_id"],
                    source_text_sha256=row["source_text_sha256"],
                )
                for row in first
            )
        )

    def test_selected_samples_are_eligible_at_both_contexts(self) -> None:
        cohort = freeze_cohort(_inventory())
        self.assertTrue(
            all(
                sample["eligible_at_both_required_contexts"]
                for task in cohort["tasks"]
                for sample in task["selected_physical_samples"]
            )
        )

    def test_same_physical_sample_is_used_at_short_and_long(self) -> None:
        cohort = freeze_cohort(_inventory())
        for task in cohort["tasks"]:
            for sample in task["selected_physical_samples"]:
                self.assertEqual(
                    set(map(int, sample["contexts"])),
                    set(task["paired_context_lengths"]),
                )
                self.assertEqual(
                    {row["sample_id"] for row in sample["contexts"].values()},
                    {sample["sample_id"]},
                )

    def test_task_inventory_removes_prior_sample_before_eligibility_output(
        self,
    ) -> None:
        prepared = []
        for index in range(2):
            source_hash = hashlib.sha256(str(index).encode()).hexdigest()
            prepared.append(
                {
                    "sample_id": f"qasper-{index}",
                    "source_sample_id": f"source-{index}",
                    "source_text_sha256": source_hash,
                    "source_index": index,
                    "document_token_length": 40000,
                    "source_full_prompt_length": 40000,
                    "fixed_prompt_length": 100,
                    "longbench_task": "qasper",
                    "context": "x" * 40000,
                    "query": "question",
                }
            )
        excluded = {
            "sample_id": {("qasper", "qasper-0")},
            "source_sample_id": set(),
            "source_text_sha256": set(),
        }
        inventory = build_task_inventory(
            task="qasper",
            source_record_count=2,
            prepared=prepared,
            tokenizer=CharacterTokenizer(),
            exclusions=excluded,
        )
        for context in inventory["contexts"]:
            self.assertNotIn("qasper-0", context["eligible_sample_ids"])

    def test_selected_samples_do_not_appear_in_exclusion_set(self) -> None:
        cohort = freeze_cohort(_inventory())
        excluded = exclusion_index(_exclusions())
        for task in cohort["tasks"]:
            for sample in task["selected_physical_samples"]:
                self.assertFalse(
                    exclusion_matches(
                        {
                            "longbench_task": task["task"],
                            "sample_id": sample["sample_id"],
                            "source_sample_id": sample["source_sample_id"],
                            "source_text_sha256": sample["source_text_sha256"],
                        },
                        excluded,
                    )
                )

    def test_primary_n16_matrix_contains_base_and_all_families(self) -> None:
        cohort = freeze_cohort(_inventory())
        matrix = build_matrix(
            cohort, node_budget=16, role="primary_same_actual_verification_work"
        )
        self.assertEqual(matrix["node_budget"], 16)
        self.assertEqual(
            {row["configuration_family"] for row in matrix["rows"]}, set(FAMILIES)
        )
        self.assertEqual(matrix["row_count"], 112)

    def test_secondary_n32_matrix_uses_exact_same_cohort_and_families(self) -> None:
        artifacts = _artifacts()
        key = lambda row: (
            row["task"],
            row["sample_id"],
            row["nominal_prompt_length"],
            row["configuration_family"],
        )
        self.assertEqual(
            {key(row) for row in artifacts["primary"]["rows"]},
            {key(row) for row in artifacts["secondary"]["rows"]},
        )
        self.assertEqual(artifacts["secondary"]["node_budget"], 32)

    def test_no_family_is_selected_or_removed(self) -> None:
        artifacts = _artifacts()
        for matrix in (artifacts["primary"], artifacts["secondary"]):
            self.assertEqual(tuple(matrix["frozen_configuration_families"]), FAMILIES)
            self.assertEqual(
                tuple(matrix["frozen_refinement_families"]),
                FROZEN_REFINEMENT_FAMILIES,
            )
            self.assertFalse(matrix["family_selected_or_removed"])

    def test_q1_excluded_and_always_fixed_verifier_frozen(self) -> None:
        artifacts = _artifacts()
        for matrix in (artifacts["primary"], artifacts["secondary"]):
            self.assertTrue(
                all(
                    row["verification_mode"] == VERIFICATION_MODE
                    and row["candidate_label_authority"] == CANDIDATE_LABEL_AUTHORITY
                    and row["candidate_label_source"] == CANDIDATE_LABEL_SOURCE
                    and not row["q1_replay_used_as_target_label_source"]
                    for row in matrix["rows"]
                )
            )

    def test_independent_task_count_excludes_govreport(self) -> None:
        cohort = freeze_cohort(_inventory())
        self.assertEqual(cohort["independent_task_count"], 2)
        self.assertEqual(tuple(cohort["independent_tasks"]), INDEPENDENT_TASKS)
        gov = next(row for row in cohort["tasks"] if row["task"] == "gov_report")
        self.assertFalse(gov["contributes_to_independent_task_count"])

    def test_task_level_rule_cannot_be_replaced_by_global_pool(self) -> None:
        rules = build_aggregation_rules()
        self.assertEqual(
            rules["aggregation_unit_for_final_stability"], "task_context_stratum"
        )
        self.assertFalse(
            rules["combined_sample_weighted_global_pool_can_replace_task_level_rule"]
        )
        self.assertEqual(
            rules["stable_cross_task_same_work_signal_rule"][
                "required_task_context_stratum_count_per_family"
            ],
            4,
        )

    def test_freeze_executes_and_authorizes_no_model_forward(self) -> None:
        artifacts = _artifacts()
        manifest = artifacts["manifest"]
        self.assertFalse(manifest["model_forward_executed"])
        self.assertFalse(manifest["target_model_loaded"])
        self.assertFalse(manifest["draft_model_loaded"])
        self.assertFalse(manifest["h2_primary_execution_authorized"])
        self.assertFalse(manifest["h2_secondary_execution_authorized"])

    def test_persisted_freeze_passes_independent_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "freeze"
            persist_freeze(output, _artifacts())
            result = check_h2_freeze(output)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["frozen_physical_sample_count"], 14)
            self.assertEqual(result["primary_matrix_row_count"], 112)
            self.assertFalse(result["h2_execution_authorized"])

    def test_ineligible_freeze_persists_inventory_and_stop_status(self) -> None:
        inventory = _inventory()
        exclusions = _exclusions()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "stopped"
            persist_ineligible_inventory(
                output,
                inventory=inventory,
                exclusions=exclusions,
                tokenizer_provenance={"model_loaded": False},
                h11_provenance={"checker_status": "accepted"},
                status="insufficient_common_long_context_eligibility",
                message="not enough samples",
            )
            manifest = json.loads(
                (output / "phase2d2a-h2-freeze-manifest.json").read_text()
            )
            self.assertEqual(
                manifest["status"],
                "insufficient_common_long_context_eligibility",
            )
            self.assertFalse(manifest["model_forward_executed"])
            self.assertTrue((output / "phase2d2a-h2-heldout-inventory.json").is_file())

    def test_import_does_not_load_torch_or_transformers(self) -> None:
        command = (
            "import sys; import scripts.freeze_phase2d2a_h2_cohort; "
            "print(int('torch' in sys.modules), int('transformers' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "0 0")

    def test_online_and_third_party_sources_are_unchanged(self) -> None:
        expected = {
            "third_party/qwen3_8b_dflash_b16/instrumented/dflash.py": (
                "6b20d1476af5746ffa511df3441230ee25cdc5f25f25d98b7eae8f72dbf8e533"
            ),
            "dflash/phase2d2/refinement.py": (
                "9c4e5acd20d4076332841d15277a54b821c2f3b88007cea53ff90410f517b9bc"
            ),
            "dflash/phase2d2/robustness.py": (
                "063b2689f63d89504ac319e3c0b46b717ff0e4d5717c7514d965043ff130cf01"
            ),
            "dflash/phase2d2/tensor.py": (
                "dec781f37c0b35fe5b546c457ae2f420fa6578281d8dce696366bb4d83d863ab"
            ),
        }
        for relative, expected_hash in expected.items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected_hash, relative)


if __name__ == "__main__":
    unittest.main()
