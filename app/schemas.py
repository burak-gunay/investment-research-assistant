"""Response contract for a cited answer, and the labelled passages it cites."""

from typing import Literal

from pydantic import BaseModel


class Passage(BaseModel):
    """One retrieved chunk, labelled with where it came from."""

    document: str
    page: int
    text: str

    def label(self) -> str:
        return f"{self.document} p{self.page}"


class Citation(BaseModel):
    document: str
    page: int


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class Answer(BaseModel):
    status: Literal["answered", "not_found"]
    summary: str
    claims: list[Claim]

    def cited_pages(self) -> set[tuple[str, int]]:
        return {
            (citation.document, citation.page)
            for claim in self.claims
            for citation in claim.citations
        }
