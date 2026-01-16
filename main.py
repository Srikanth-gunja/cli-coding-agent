from client.llm_client import LLMClinet
import asyncio

async def main():
    client=LLMClinet()
    messages=[{
        "role":"user",
        "content":"Hello "
    }]
    async for event in client.chat_completion(messages):
        print(event)
asyncio.run(main=main())
