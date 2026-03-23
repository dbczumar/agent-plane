# runtime

The execution engine — how an agent **runs**. Given a spec and user input, it drives the agent's reasoning loop: invoking LLMs, calling tools, managing skills, and producing responses.

The runtime is a library, not a service. The server is its primary host, but it can also be used directly for local development, embedded in other applications, or invoked from tests.
