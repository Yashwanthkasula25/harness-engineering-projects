"""Case-facts extraction into a persistent block at the top of context.

Extraction is LLM-driven: one Claude call against the full transcript that returns
strict JSON for the 12 required fields. This is commonly called a *scratchpad* —
same concept, different word: a dense structured block that survives compression
and is placed at the top boundary of context so the model can recover
transactional facts without scanning thousands of tokens of narrative.

Missing-field behavior raises `CaseFactExtractionError` listing the gaps —
silent null-fill is forbidden.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from retail_context.client import complete_with_system, get_model
from retail_context.transcript import Transcript


REQUIRED_FIELDS: tuple[str, ...] = (
    "customer_id",
    "refund_order_id",
    "refund_amount_usd",
    "refund_status",
    "subscription_id",
    "subscription_plan",
    "subscription_cancel_reason",
    "subscription_status",
    "active_payment_method_last4",
    "new_payment_method_last4",
    "payment_update_failure_code",
    "payment_update_status",
)


@dataclass
class CaseFacts:
    customer_id: str
    refund_order_id: str
    refund_amount_usd: float
    refund_status: str
    subscription_id: str
    subscription_plan: str
    subscription_cancel_reason: str
    subscription_status: str
    active_payment_method_last4: str
    new_payment_method_last4: str
    payment_update_failure_code: str
    payment_update_status: str

    def to_markdown(self) -> str:
        return (
            "# Case Facts\n\n"
            "**Customer**\n"
            f"- customer_id: {self.customer_id}\n\n"
            "**Refund (resolved)**\n"
            f"- refund_order_id: {self.refund_order_id}\n"
            f"- refund_amount_usd: {self.refund_amount_usd}\n"
            f"- refund_status: {self.refund_status}\n\n"
            "**Subscription (resolved)**\n"
            f"- subscription_id: {self.subscription_id}\n"
            f"- subscription_plan: {self.subscription_plan}\n"
            f"- subscription_cancel_reason: {self.subscription_cancel_reason}\n"
            f"- subscription_status: {self.subscription_status}\n\n"
            "**Payment update (active)**\n"
            f"- active_payment_method_last4: {self.active_payment_method_last4}\n"
            f"- new_payment_method_last4: {self.new_payment_method_last4}\n"
            f"- payment_update_failure_code: {self.payment_update_failure_code}\n"
            f"- payment_update_status: {self.payment_update_status}\n"
        )


class CaseFactExtractionError(ValueError):
    def __init__(self, missing: list[str], raw: dict[str, Any]):
        super().__init__(f"case-facts extraction missing required fields: {missing}")
        self.missing = missing
        self.raw = raw


_SYSTEM_PROMPT = f"""
Extract the case facts from the provided transcript.

Return EXACTLY one JSON object with these 12 keys:

{json.dumps(REQUIRED_FIELDS)}

Requirements:
- Every required key must be present.
- Use the exact field names shown above.
- `refund_amount_usd` must be a JSON number.
- All IDs, status tokens, and payment-method last4 values must be strings.
- Preserve status tokens verbatim from the transcript.
- Last4 values must be zero-padded strings when applicable.
- If a required fact is missing from the transcript, use null.
- DO NOT invent or infer missing values.
- Output JSON only.
- No prose.
- No Markdown.
- No code fences.
"""


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.rstrip().endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def extract(
    transcript: Transcript,
    *,
    model: str | None = None,
    log_path: Path | None = None,
) -> CaseFacts:
    user = f"Transcript:\n\n{transcript.full_text}"

    text, input_tokens, output_tokens = complete_with_system(
        _SYSTEM_PROMPT,
        user,
        model=model,
        max_tokens=2048,
    )

    parsed = _parse_json(text)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "model": model or get_model(),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "raw": parsed,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    missing = [
        name
        for name in REQUIRED_FIELDS
        if name not in parsed or parsed[name] is None or parsed[name] == ""
    ]

    if missing:
        raise CaseFactExtractionError(missing=missing, raw=parsed)

    return CaseFacts(
        customer_id=str(parsed["customer_id"]),
        refund_order_id=str(parsed["refund_order_id"]),
        refund_amount_usd=float(parsed["refund_amount_usd"]),
        refund_status=str(parsed["refund_status"]),
        subscription_id=str(parsed["subscription_id"]),
        subscription_plan=str(parsed["subscription_plan"]),
        subscription_cancel_reason=str(parsed["subscription_cancel_reason"]),
        subscription_status=str(parsed["subscription_status"]),
        active_payment_method_last4=str(parsed["active_payment_method_last4"]),
        new_payment_method_last4=str(parsed["new_payment_method_last4"]),
        payment_update_failure_code=str(parsed["payment_update_failure_code"]),
        payment_update_status=str(parsed["payment_update_status"]),
    )


def to_dict(facts: CaseFacts) -> dict[str, Any]:
    return asdict(facts)