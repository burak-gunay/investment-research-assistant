"""One answering call: labelled passages in, a validated cited answer out."""

import anthropic
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import Answer, Passage

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
MAX_ATTEMPTS = 2

SYSTEM = """You answer questions about company reports using only the passages provided.

Rules:
- Break your answer into claims. Each claim is one factual statement.
- Every claim must cite at least one passage, by its exact document name and page number.
- Never cite a document or page that is not in the passages given to you.
- If the passages do not support an answer, set status to "not_found", leave claims
  empty, and say in the summary what is missing.
- The summary is the prose answer a reader sees. It must say only what the claims say."""


class GroundingError(ValueError):
    """The response is well-formed but makes claims the passages do not support."""


class AnswerUnavailable(RuntimeError):
    """No usable answer after every attempt."""


def _build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = [f"[{p.label()}]\n{p.text}" for p in passages]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def check_grounding(answer: Answer, passages: list[Passage]) -> None:
    """Enforce the rules the schema cannot express."""
    if answer.status == "answered" and not answer.claims:
        raise GroundingError("status is answered but there are no claims")
    if answer.status == "not_found" and answer.claims:
        raise GroundingError("status is not_found but claims were returned")

    for index, claim in enumerate(answer.claims):
        if not claim.citations:
            raise GroundingError(f"claim {index} has no citations")

    available = {(p.document, p.page) for p in passages}
    invented = answer.cited_pages() - available
    if invented:
        cited = ", ".join(f"{doc} p{page}" for doc, page in sorted(invented))
        raise GroundingError(f"cited passages that were not provided: {cited}")


def ask(
    question: str,
    passages: list[Passage],
    *,
    client: anthropic.Anthropic | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> Answer:
    settings = get_settings()
    client = client or anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value()
    )

    messages: list[dict] = [
        {"role": "user", "content": _build_prompt(question, passages)}
    ]
    last_error: Exception | None = None

    for _ in range(max_attempts):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=messages,
            output_format=Answer,
        )

        if response.stop_reason == "refusal":
            raise AnswerUnavailable(
                f"model refused: {getattr(response.stop_details, 'category', None)}"
            )
        if response.stop_reason == "max_tokens":
            raise AnswerUnavailable("response hit max_tokens and was truncated")

        raw = next((b.text for b in response.content if b.type == "text"), "")

        try:
            answer = response.parsed_output
            if answer is None:
                raise ValueError("no parsed output in response")
            check_grounding(answer, passages)
            return answer
        except (ValidationError, ValueError) as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"That response was rejected: {exc}\n"
                        "Return a corrected response in the same format."
                    ),
                },
            ]

    raise AnswerUnavailable(
        f"no valid answer after {max_attempts} attempts: {last_error}"
    )


if __name__ == "__main__":
    sample = [
        Passage(
            document="sample-annual-report.pdf",
            page=12,
            text=(
                "Our operations depend on a concentrated set of component suppliers "
                "in two regions. Disruption at any one of them would delay shipments."
            ),
        ),
        Passage(
            document="sample-annual-report.pdf",
            page=48,
            text="Revenue for the year was 412.6 million, up 8% year over year.",
        ),
    ]
    for question in ("What supply chain risks are disclosed?", "Who is the auditor?"):
        result = ask(question, sample)
        print(f"\n{question}\n  status: {result.status}\n  summary: {result.summary}")
        for claim in result.claims:
            pages = ", ".join(f"{c.document} p{c.page}" for c in claim.citations)
            print(f"  claim: {claim.text}\n    cited: {pages}")
