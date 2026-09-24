"""Deterministic tool-output pruning for the verbose `lookup_order` response.

The "Tool Context Pruning" pattern: application-side filtering
of a verbose tool result so only the fields needed for the immediate decision survive
into context. For return/refund reasoning, exactly five fields matter — order identity,
when it was placed, what it cost, whether it shipped, and the return-window deadline.

Why each kept field is the only one that matters for return/refund reasoning:
  - order_id: Identifies the order being evaluated for the return/refund decision.
  - order_date: Establishes when the order was placed and supports return-window reasoning.
  - order_total_usd: Establishes the monetary value relevant to the refund decision.
  - fulfillment_status: Shows whether the order was shipped, which affects return/refund handling.
  - return_eligible_until: Directly determines whether the order is still within the return window.

Implementation: deterministic field selection (no LLM call). The pruner has no
`anthropic` import — enforced by an AST audit.
"""
from __future__ import annotations

# The exact 5 fields returned, in output order.
KEPT_FIELDS: tuple[str, ...] = (
    "order_id",
    "order_date",
    "order_total_usd",
    "fulfillment_status",
    "return_eligible_until",
)


class PrunerMissingFieldError(KeyError):
    """Raised when the raw tool response is missing one of the required kept fields."""


def prune_lookup_order(raw: dict) -> dict:
    missing = [field for field in KEPT_FIELDS if field not in raw]

    if missing:
        raise PrunerMissingFieldError(
            f"Missing required fields: {', '.join(missing)}"
        )

    return {field: raw[field] for field in KEPT_FIELDS}