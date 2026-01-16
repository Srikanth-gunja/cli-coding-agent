from openai import AsyncOpenAI
import asyncio

# client = AsyncOpenAI(
#     api_key="sk-or-v1-5c1334e6a2d9df5f19d29eb0575b8e735fbf129038d1dd67bd233dba",
#     base_url="https://openrouter.ai/api/v1",
# )


async def main():
    res = await client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {
                "role": "user",
                "content": "Hello give me ten lines about ai agnets ",
            }
        ],
        stream=True,
    )
    
        
    async for chunk in res:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)


asyncio.run(main())
