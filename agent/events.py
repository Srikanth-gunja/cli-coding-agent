from __future__ import annotations
from enum import Enum
from typing import Any
from client.response import TokenUsage
from pydantic import BaseModel, Field


class AgentEventType(str, Enum):

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"


class AgentEvent(BaseModel):
    type: AgentEventType
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        return cls(type=AgentEventType.AGENT_START, data={"message": message})

    @classmethod
    def agent_error(
        cls, error: str, details: dict[str, Any] | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error, "details": details or {}},
        )

    @classmethod
    def agent_end(
        cls, response: str | None = None, usage: TokenUsage | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_END,
            data={"response": response, "usage": usage.__dict__ if usage else None},
        )

    @classmethod
    def text_delta(cls, content: str) -> AgentEvent:
        return cls(type=AgentEventType.TEXT_DELTA, data={"content": content})

    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        return cls(type=AgentEventType.TEXT_COMPLETE, data={"content": content})
