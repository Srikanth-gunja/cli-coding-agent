import sys
from agent.agent import Agent
from agent.events import AgentEventType
from client.llm_client import LLMClinet
import asyncio
import click
from typing import Any
from ui.tui import TUI


class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI()

    async def run_single(self, message: str):
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str):
        if not self.agent:
            return None
        assistant_streaming = False
        async for event in self.agent.run(message=message):
            if event.type == AgentEventType.TEXT_DELTA:
                if not assistant_streaming:
                    self.tui.begin_assiant()
                    assistant_streaming = True
                content = event.data.get("content", "")
                self.tui.stream_assiant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                self.tui.end_assistant()

            elif event.type == AgentEventType.AGENT_ERROR:
                # print("YO error cathed")
                self.tui.error_message(f"{event.data.get("error","Unknown error")}")


# async def run(messages):
#     client=LLMClinet()
#     async for event in client.chat_completion(messages):
#         print(event)


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None):
    print(prompt)
    cli = CLI()

    result = asyncio.run(main=cli.run_single(prompt or ""))
    if result is None:
        sys.exit(1)


main()
