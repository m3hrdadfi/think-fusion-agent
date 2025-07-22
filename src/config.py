import os
from enum import Enum
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


class Config(BaseModel):
    """The configurable fields for the research assistant."""

    max_web_research_attempts: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform"
    )
    num_top_relevant_chunks: int = Field(
        default=5,
        title="Num Relevant Chunks",
        description="Number of top relevant chunks to consider"
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use"
    )
    local_em: str = Field(
        default="nomic-embed-text:v1.5",
        title="Embedding Model Name",
        description="Name of the LLM model to use"
    )
    llm_provider: Literal["ollama", "lmstudio"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (Ollama or LMStudio)"
    )
    search_api: Literal["perplexity", "tavily", "duckduckgo", "searxng"] = Field(
        default="duckduckgo",
        title="Search API",
        description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API"
    )
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        title="LMStudio Base URL",
        description="Base URL for LMStudio OpenAI-compatible API"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Config":
        """Create a Config instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
