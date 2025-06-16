# app/services/llm_service.py
import os
import openai
from typing import Optional


class LLMService:
    def __init__(self):
        vllm_host = os.getenv("VLLM_HOST", "localhost")
        vllm_port = int(os.getenv("VLLM_PORT", 8000))

        self.client = openai.AsyncOpenAI(
            base_url=f"http://{vllm_host}:{vllm_port}/v1",
            api_key="fake-key"  # vLLM doesn't require real API key
        )

    async def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7
    ) -> str:
        """Generate text using the LLM"""

        response = await self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",  # This should match your model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content