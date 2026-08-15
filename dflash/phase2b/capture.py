"""Pure-Python alignment of Phase 2B captures with verified output tokens."""

from __future__ import annotations

from itertools import pairwise
from typing import Any


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    """Return the number of equal leading token IDs."""

    length = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        length += 1
    return length


def _alignment_error(
    *,
    round_index: int,
    position: int,
    draft_ids: list[int],
    reference_ids: list[int],
    root_offset: int,
    accepted: int,
    observed_lcp: int,
) -> RuntimeError:
    return RuntimeError(
        "Phase 2B final-output alignment invariant failed: "
        f"round={round_index}, position={position}, accepted={accepted}, "
        f"observed_lcp={observed_lcp}, root_output_offset={root_offset}, "
        f"draft={draft_ids}, reference={reference_ids}"
    )


def build_capture_round_rows(
    *,
    diagnostics: dict[str, Any],
    final_verified_generated_token_ids: list[int],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach each exact-logit candidate block to its final verified continuation.

    An uncensored rejection is exactly observable because the correction token is
    present after the accepted prefix. If EOS or the output budget ends at or
    inside the accepted prefix, only consistency of the observable prefix can be
    asserted; the record is explicitly right-censored.
    """

    if not diagnostics.get("tree_candidate_capture", {}).get("enabled"):
        raise ValueError("diagnostics do not contain enabled Phase 2B capture")
    proposed_count = int(diagnostics["draft_block_size"]) - 1
    if proposed_count != 15:
        raise ValueError("Phase 2B requires the pinned 15-token draft block")

    rows = []
    for accounting in diagnostics["rounds"]:
        capture = accounting.get("tree_candidate_capture")
        if not capture:
            raise ValueError("a Phase 2B diagnostic round lacks candidate capture")
        round_index = int(accounting["round_index"])
        root_offset = int(capture["round_output_start_offset"])
        if not 0 <= root_offset < len(final_verified_generated_token_ids):
            raise _alignment_error(
                round_index=round_index,
                position=0,
                draft_ids=capture["baseline_draft_token_ids"],
                reference_ids=[],
                root_offset=root_offset,
                accepted=int(accounting["accepted_draft_token_count"]),
                observed_lcp=0,
            )
        observed_root = int(final_verified_generated_token_ids[root_offset])
        if observed_root != int(capture["root_token_id"]):
            raise _alignment_error(
                round_index=round_index,
                position=0,
                draft_ids=capture["baseline_draft_token_ids"],
                reference_ids=[observed_root],
                root_offset=root_offset,
                accepted=int(accounting["accepted_draft_token_count"]),
                observed_lcp=0,
            )

        reference = [
            int(value)
            for value in final_verified_generated_token_ids[
                root_offset + 1 : root_offset + 1 + proposed_count
            ]
        ]
        draft_ids = [int(value) for value in capture["baseline_draft_token_ids"]]
        accepted = int(accounting["accepted_draft_token_count"])
        observed_lcp = longest_common_prefix(draft_ids, reference)
        observed_depth = len(reference)
        right_censored = observed_depth < proposed_count
        exactly_observable = observed_depth > accepted or (
            accepted == proposed_count and observed_depth == proposed_count
        )
        if exactly_observable:
            if observed_lcp != accepted:
                mismatch_position = min(observed_lcp, observed_depth) + 1
                raise _alignment_error(
                    round_index=round_index,
                    position=mismatch_position,
                    draft_ids=draft_ids,
                    reference_ids=reference,
                    root_offset=root_offset,
                    accepted=accepted,
                    observed_lcp=observed_lcp,
                )
            alignment_status = (
                "exact_lcp_equals_accepted_right_censored_after_mismatch"
                if right_censored
                else "exact_lcp_equals_accepted"
            )
        else:
            if observed_depth > accepted or observed_lcp != observed_depth:
                raise _alignment_error(
                    round_index=round_index,
                    position=observed_lcp + 1,
                    draft_ids=draft_ids,
                    reference_ids=reference,
                    root_offset=root_offset,
                    accepted=accepted,
                    observed_lcp=observed_lcp,
                )
            alignment_status = "right_censored_observed_prefix_consistent"

        row = {
            "schema_version": "1.0",
            **metadata,
            "round_index": round_index,
            "context_length_before": int(accounting["context_length_before"]),
            "context_length_after": int(accounting["context_length_after"]),
            "root_absolute_output_index": int(
                capture["root_absolute_output_index"]
            ),
            "round_output_start_offset": root_offset,
            "root_token_id": int(capture["root_token_id"]),
            "baseline_draft_token_ids": draft_ids,
            "positions": capture["positions"],
            "top_k": int(capture["top_k"]),
            "proposed_draft_token_count": proposed_count,
            "accepted_draft_token_count": accepted,
            "rejected_draft_token_count": int(
                accounting["rejected_draft_token_count"]
            ),
            "committed_token_count": int(accounting["committed_token_count"]),
            "raw_cumulative_generated_token_count": int(
                accounting["raw_cumulative_generated_token_count"]
            ),
            "cumulative_returned_token_count": int(
                accounting["cumulative_generated_token_count"]
            ),
            "returned_token_delta": int(accounting["returned_token_delta"]),
            "stop_reason": accounting["stop_reason"],
            "truncated_token_count": int(accounting["truncated_token_count"]),
            "terminal_round": accounting["stop_reason"] != "continue",
            "final_verified_output_token_ids": [
                int(value) for value in final_verified_generated_token_ids
            ],
            "final_verified_output_scope": "generated_tokens_only",
            "reference_target_continuation_token_ids": reference,
            "reference_observed_depth": observed_depth,
            "reference_right_censored": right_censored,
            "full_depth_evaluation_eligible": observed_depth == proposed_count,
            "censor_reason": diagnostics["stop_reason"] if right_censored else None,
            "alignment": {
                "status": alignment_status,
                "final_output_lcp": observed_lcp,
                "accepted_draft_token_count": accepted,
                "exact_lcp_invariant_evaluable": exactly_observable,
                "exact_lcp_invariant_holds": (
                    observed_lcp == accepted if exactly_observable else None
                ),
                "immediate_verification_lcp": int(
                    capture["immediate_verification_lcp"]
                ),
                "immediate_verification_invariant_holds": bool(
                    capture["immediate_alignment_verified"]
                ),
            },
            "logit_source": capture["logit_source"],
            "score_semantics": capture["score_semantics"],
            "tie_break_rule": capture["tie_break_rule"],
            "full_vocabulary_logits_saved": False,
            "target_logits_saved": False,
            "model_hidden_states_saved": False,
            "kv_cache_saved": False,
        }
        rows.append(row)
    return rows


def validate_capture_round(row: dict[str, Any]) -> None:
    """Validate structural and alignment invariants without model imports."""

    proposed = int(row["proposed_draft_token_count"])
    if proposed != 15 or len(row["positions"]) != proposed:
        raise ValueError("capture must contain exactly 15 draft positions")
    baseline = [int(value) for value in row["baseline_draft_token_ids"]]
    if len(baseline) != proposed:
        raise ValueError("baseline draft chain must contain 15 tokens")
    top_k = int(row["top_k"])
    for index, position in enumerate(row["positions"], 1):
        if int(position["block_position"]) != index:
            raise ValueError("capture positions must be consecutive and one-based")
        candidates = position["candidates"]
        if len(candidates) != top_k:
            raise ValueError("every capture position must contain top_k candidates")
        if [int(item["rank"]) for item in candidates] != list(
            range(1, top_k + 1)
        ):
            raise ValueError("candidate ranks must be consecutive and one-based")
        token_ids = [int(item["token_id"]) for item in candidates]
        if len(set(token_ids)) != top_k:
            raise ValueError("candidate token IDs must be unique at each position")
        if token_ids[0] != baseline[index - 1]:
            raise ValueError("baseline draft token must equal captured Top-1")
        scores = [float(item["log_probability"]) for item in candidates]
        if any(left < right for left, right in pairwise(scores)):
            raise ValueError("candidate log probabilities must be non-increasing")
        for left, right, left_id, right_id in zip(
            scores, scores[1:], token_ids, token_ids[1:]
        ):
            if left == right and left_id > right_id:
                raise ValueError("equal-score candidates must use ascending token IDs")

    reference = [int(value) for value in row["reference_target_continuation_token_ids"]]
    lcp = longest_common_prefix(baseline, reference)
    if lcp != int(row["alignment"]["final_output_lcp"]):
        raise ValueError("stored final-output LCP is inconsistent")
    if row["alignment"]["exact_lcp_invariant_evaluable"]:
        if lcp != int(row["accepted_draft_token_count"]):
            raise ValueError("Top-1 LCP must equal accepted drafts when observable")
    elif lcp != len(reference) or len(reference) > int(
        row["accepted_draft_token_count"]
    ):
        raise ValueError("right-censored reference must be a matching accepted prefix")
