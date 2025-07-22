import operator
from dataclasses import dataclass, field

from typing_extensions import Annotated


@dataclass
class SummaryState:
    """Holds intermediate and final values for the PDF-Web research pipeline."""

    user_question: str = field(default=None)
    pdf_path: str = field(default=None)
    extracted_text: str = field(default=None)

    relevant_chunks: Annotated[list, operator.add] = field(default_factory=list)

    search_query: str = field(default=None)
    web_context: Annotated[list, operator.add] = field(default_factory=list)
    web_sources: Annotated[list, operator.add] = field(default_factory=list)
    search_action: str = field(default=None)
    search_attempts: int = field(default=0)
    web_context_empty: bool = field(default=False)

    final_answer: str = field(default=None)
    answer_history: Annotated[list[str], operator.add] = field(default_factory=list)
    has_converged: bool = field(default=False)
    final_answer_bundle: dict = field(default_factory=dict)
    answer_is_sufficient: bool = field(default=False)
    answer_reflection_reasoning: str = field(default="")

    post_final_reflection: str = field(default="")


@dataclass
class SummaryStateInput:
    """Initial input for the pipeline."""

    user_question: str = field(default=None)
    pdf_path: str = field(default=None)


@dataclass
class SummaryStateOutput:
    """Final output from the pipeline."""

    final_answer: str = field(default=None)
    post_final_reflection: str = field(default=None)
    final_answer_bundle: dict = field(default_factory=dict)
