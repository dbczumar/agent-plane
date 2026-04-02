"""
Multi-provider LLM client with OpenAI Responses API interface.

Usage::

    from agent_plane.llms import Client

    client = Client()
    resp = client.responses.create(
        input=[{"role": "user", "content": "Hello"}],
        instructions="You are a helpful assistant.",
        model="anthropic/claude-sonnet-4-20250514",
    )
"""

from agent_plane.llms.client import Client

__all__ = ["Client"]
