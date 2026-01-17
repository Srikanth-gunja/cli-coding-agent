from typing import Any, AsyncGenerator
from agent.events import AgentEvent, AgentEventType
import client
from client.llm_client import LLMClinet
from client.response import StreamEventType
from context.manager import ContextManager


class Agent:
    def __init__(self):
        self.client = LLMClinet()
        self.context_manager = ContextManager()

    async def run(self, message: str):

        yield AgentEvent.agent_start(message="Agent Started")
        self.context_manager.add_user_message(message)
        final_response = ""
        async for event in self._agentic_loop():
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
            yield event

        yield AgentEvent.agent_end(response=final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        messages = self.context_manager.get_messages()

        response_text = ""
        async for event in self.client.chat_completion(messages):

            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content=content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(error=event.error or "Unknown Error")
        self.context_manager.add_assistant_message(response_text or "")
        if response_text:

            yield AgentEvent.text_complete(content=response_text)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
            self.client = None
