import asyncio
from typing import Any, AsyncGenerator
from openai import APIError, AsyncOpenAI, RateLimitError 
import os
from .events import (
    TokenUsage,
    StreamEventType,
    EventType,
    TextDelta
)
from dotenv import load_dotenv
load_dotenv()

class LLMClinet:
    def __init__(self):
        self._client=None 
        self._max_tries:int=3

    def get_client(self):
        if self._client is None:
            self._client=AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
       
        return self._client
            

    async def chat_completion(self,messages:list[dict[str,Any]],stream=True)->AsyncGenerator[StreamEventType,None]:
        client=self.get_client()
    
        kwargs = {"model": "nvidia/nemotron-3-nano-30b-a3b:free", "messages": messages,"stream":stream}
        
        
        for attempt in range(self._max_tries+1):
            try:
                if stream:
                    event = self._stream_response(client,kwargs)
                    async for e in event:
                        yield e
                else:
                    event = await   self._non_stream_response(client,kwargs)
                    yield event
                return
            except RateLimitError as ae:
                if attempt <self._max_tries:
                    wait_time=2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEventType(
                        type=EventType.ERROR,
                        error=f"Rate Limit exceede :{ae}"
                    )
                    return 
            except APIError as ae:
                    yield StreamEventType(
                        type=EventType.ERROR,
                        error=f"API Error:{ae}"
                    )
                    return 

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict
    ) -> AsyncGenerator[StreamEventType, None]:

        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in response:
           
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            delta = chunk.choices[0].delta

            if delta and delta.content:
                yield StreamEventType(
                    type=EventType.TEXT_DELTA,
                    text_delta=TextDelta(content=delta.content),
                )

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        yield StreamEventType(
            type=EventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )
              
        
        
    
    async def _non_stream_response(self,client,kwargs)->StreamEventType:
      
        response= await client.chat.completions.create(**kwargs)
        
        choices=response.choices[0]
        messages=choices.message
        
        text_delta=None
        usage=None 
        if messages.content:
            text_delta=TextDelta(content=messages.content)
        if response.usage:
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEventType(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choices.finish_reason,
            usage=usage
        )
        
        

