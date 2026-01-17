from typing import Any, Dict
from prompts.system import get_system_prompt
from pydantic import BaseModel

from utils.text import count_tokens


class MessageItem(BaseModel):
    role: str
    content: str
    token_usage: int | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"role": self.role}

        if self.content:
            res["content"] = self.content
        return res


class ContextManager:
    def __init__(self) -> None:
        self._system_prompt = get_system_prompt()
        self._message_item: list[MessageItem] = []
        self._model_name: str = "nvidia/nemotron-3-nano-30b-a3b:free"

    def add_user_message(self, message: str):
        msg = MessageItem(
            role="user",
            content=message,
            token_usage=count_tokens(message, self._model_name),
        )
        self._message_item.append(msg)

    def add_assistant_message(self, message: str):
        msg = MessageItem(
            role="user",
            content=message,
            token_usage=count_tokens(message, self._model_name),
        )
        self._message_item.append(msg)

    def get_messages(self) -> list[dict[str, Any]]:
        msg = []
        msg.append({"role": "system", "content": self._system_prompt})
        for item in self._message_item:
            msg.append(item.to_dict())
        return msg
