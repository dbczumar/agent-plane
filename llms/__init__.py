"""
Multi-provider LLM client with OpenAI Responses API interface.

Usage::

    from llms import Client

    client = Client()
    resp = client.responses.create(
        input=[{"role": "user", "content": "Hello"}],
        instructions="You are a helpful assistant.",
        model="anthropic/claude-sonnet-4-20250514",
    )
"""

from llms.client import Client

__all__ = ["Client"]
