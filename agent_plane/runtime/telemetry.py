"""
Agent-plane observability on top of the MLflow Tracing SDK.

See ``designs/OBSERVABILITY.md`` for the full design. The module
is intentionally thin — it holds only the agent-plane-specific
concerns:

* **Trace ID derivation from the response ID.** Agent-plane response
  IDs are ``resp_<32-char hex>``. We reuse the hex suffix as the
  W3C trace ID so operators can look up a trace by its response ID
  without a lookup table. :func:`trace_context_for_response` wraps
  MLflow's public distributed-tracing entry point.

* **Runtime init.** :func:`init` flips
  ``MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false`` so MLflow shares the
  global ``TracerProvider`` with raw OTel instrumentation
  (FastAPI / HTTPX) and flips ``MLFLOW_ENABLE_OTLP_EXPORTER=true``
  when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set — vendor-neutral OTLP
  export by default.

* **Subprocess trace propagation.** :func:`get_traceparent_env`
  serializes the current trace context into env vars the executor
  subprocess launchers can merge into their child process env.

* **A handful of record helpers** where the work is non-trivial
  (LLM usage normalization, cancellation tagging). Trivial
  operations like ``span.set_attribute(...)`` are called directly
  at instrumentation sites.

Call sites import this module for init + the trace-context wrapper,
and otherwise call ``mlflow`` / ``mlflow.entities.SpanType`` /
``span.set_inputs`` / ``span.set_outputs`` directly.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from mlflow.entities.span import LiveSpan

_logger = logging.getLogger(__name__)

_RESP_PREFIX = "resp_"
_HEX_LEN = 32
_DUMMY_PARENT_SPAN_ID = "1000000000000001"
_W3C_VERSION = "00"
_W3C_FLAGS_SAMPLED = "01"

_capture_content: bool = False
_initialized: bool = False


def _env_bool(name: str) -> bool:
    """
    Parse a boolean environment variable.

    Truthy values are ``"true"``, ``"1"``, ``"yes"`` (case-insensitive).
    Anything else (including unset) is ``False``.

    :param name: The environment variable name, e.g.
        ``"AGENT_PLANE_OTEL_CAPTURE_CONTENT"``.
    :returns: ``True`` if the env var is set to a truthy value.
    """
    return os.environ.get(name, "").strip().lower() in ("true", "1", "yes")


def should_capture_content() -> bool:
    """
    Return whether message content should be included on spans.

    Controlled by ``AGENT_PLANE_OTEL_CAPTURE_CONTENT``. Call sites
    read this flag before populating ``span.set_inputs`` /
    ``set_outputs`` with user messages or tool results. Content
    capture is off by default because messages may contain PII or
    secrets.

    :returns: ``True`` when content capture is enabled.
    """
    return _capture_content


def parse_provider_name(model: str) -> tuple[str, str]:
    """
    Split a provider-prefixed model string into ``(provider, model)``.

    Agent-plane model strings follow ``"<provider>/<model>"``, e.g.
    ``"openai/gpt-5.4"`` becomes ``("openai", "gpt-5.4")``. Unprefixed
    strings return an empty provider string so the span always has a
    value to record.

    :param model: The model identifier, e.g. ``"openai/gpt-5.4"``
        or ``"gpt-5.4"``.
    :returns: ``(provider, model)`` tuple. Provider is empty if the
        input has no prefix.
    """
    if "/" in model:
        provider, _, rest = model.partition("/")
        return provider, rest
    return "", model


def trace_id_from_response_id(response_id: str) -> str:
    """
    Extract the 32-char hex trace ID from an agent-plane response ID.

    Response IDs have the format ``resp_<32-char hex>`` (generated
    via ``generate_task_id``). The hex suffix is a valid 128-bit
    W3C trace ID. Reusing it as the trace ID lets operators jump
    from a response ID to its trace by stripping the ``resp_``
    prefix — no lookup table, no search query.

    :param response_id: The response/task ID, e.g.
        ``"resp_d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3"``.
    :returns: The 32-char lowercase hex trace ID.
    :raises ValueError: If the response ID does not start with
        ``"resp_"`` or the hex suffix is not exactly 32 chars.
    """
    if not response_id.startswith(_RESP_PREFIX):
        raise ValueError(
            f"Expected {_RESP_PREFIX!r} prefix, got {response_id!r}"
        )
    hex_part = response_id[len(_RESP_PREFIX) :]
    if len(hex_part) != _HEX_LEN:
        raise ValueError(
            f"Expected {_HEX_LEN} hex chars after prefix, "
            f"got {len(hex_part)} in {response_id!r}"
        )
    try:
        int(hex_part, 16)
    except ValueError as exc:
        raise ValueError(
            f"Invalid hex suffix in {response_id!r}: {exc}"
        ) from exc
    return hex_part


@contextmanager
def trace_context_for_response(
    response_id: str,
    *,
    root_response_id: str | None = None,
) -> Iterator[None]:
    """
    Set the active trace context for a workflow invocation.

    Derives the W3C trace ID from ``root_response_id`` (if set) or
    ``response_id``, then calls MLflow's public distributed-tracing
    API to register the trace and make it current. Any span started
    inside the context manager inherits this trace ID.

    For root invocations pass only ``response_id``; the trace ID is
    derived from it so direct response-ID → trace-ID lookup works.
    For sub-agent invocations pass both ``response_id`` (the
    sub-agent's own ID, exposed as ``task.id`` on the span) and
    ``root_response_id`` (the root of the spawn tree, used as the
    trace ID) so all sub-agents share the root's trace.

    :param response_id: The response/task ID for this invocation,
        e.g. ``"resp_d8e9f0a1..."``.
    :param root_response_id: The root response ID if this is a
        sub-agent invocation, otherwise ``None``.
    :raises ValueError: If ``response_id`` (or ``root_response_id``
        when set) cannot be parsed.
    """
    from mlflow.tracing.distributed import (
        set_tracing_context_from_http_request_headers,
    )

    effective = root_response_id or response_id
    trace_id_hex = trace_id_from_response_id(effective)
    traceparent = (
        f"{_W3C_VERSION}-{trace_id_hex}-"
        f"{_DUMMY_PARENT_SPAN_ID}-{_W3C_FLAGS_SAMPLED}"
    )
    with set_tracing_context_from_http_request_headers(
        {"traceparent": traceparent}
    ):
        yield


def record_llm_usage(span: "LiveSpan", usage: dict[str, Any]) -> None:
    """
    Record token usage on an LLM span.

    MLflow stores usage as a single JSON dict under
    ``mlflow.chat.tokenUsage`` and translates each field to the
    corresponding ``gen_ai.usage.*`` attribute on OTLP export —
    ``input_tokens``, ``output_tokens``, ``total_tokens``, plus
    optional cache fields.

    Cache breakdown attributes are recorded only when present.
    Their absence is meaningful (the provider did not report
    caching) and should not be masked with invented zeros.

    :param span: The LLM span to annotate.
    :param usage: Token usage dict from the LLM response. Known
        keys: ``"input_tokens"``, ``"output_tokens"``,
        ``"total_tokens"``, ``"cache_read_input_tokens"``,
        ``"cache_creation_input_tokens"``.
    """
    from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

    payload: dict[str, int] = {
        TokenUsageKey.INPUT_TOKENS: int(usage.get("input_tokens", 0)),
        TokenUsageKey.OUTPUT_TOKENS: int(usage.get("output_tokens", 0)),
    }
    total = usage.get("total_tokens")
    if total is None:
        total = (
            payload[TokenUsageKey.INPUT_TOKENS]
            + payload[TokenUsageKey.OUTPUT_TOKENS]
        )
    payload[TokenUsageKey.TOTAL_TOKENS] = int(total)
    if "cache_read_input_tokens" in usage:
        payload[TokenUsageKey.CACHE_READ_INPUT_TOKENS] = int(
            usage["cache_read_input_tokens"]
        )
    if "cache_creation_input_tokens" in usage:
        payload[TokenUsageKey.CACHE_CREATION_INPUT_TOKENS] = int(
            usage["cache_creation_input_tokens"]
        )
    span.set_attribute(SpanAttributeKey.CHAT_USAGE, payload)


def record_error(span: "LiveSpan", exc: BaseException) -> None:
    """
    Mark a span as failed with an ``error.type`` attribute.

    MLflow's ``span.record_exception`` already captures the stack
    trace and message; this helper adds the ``error.type``
    attribute (exception class name) so operators can filter by
    class in the trace backend without reading the exception event.

    :param span: The span to mark as failed.
    :param exc: The exception that caused the failure.
    """
    from mlflow.entities.span_status import SpanStatusCode

    span.set_status(SpanStatusCode.ERROR)
    span.set_attribute("error.type", type(exc).__name__)
    span.set_attribute("error.message", str(exc))
    span.record_exception(exc)


def record_cancellation(span: "LiveSpan") -> None:
    """
    Mark a span as cancelled.

    Neither OTel nor MLflow has a dedicated ``CANCELLED`` status, so
    we use ``ERROR`` with ``error.type = "cancelled"`` as the
    distinguishing attribute. Operators filter cancelled traces via
    the attribute.

    :param span: The span to mark as cancelled.
    """
    from mlflow.entities.span_status import SpanStatusCode

    span.set_status(SpanStatusCode.ERROR)
    span.set_attribute("error.type", "cancelled")


def get_traceparent_env() -> dict[str, str]:
    """
    Serialize the current trace context into env vars for subprocess
    inheritance.

    Used by executor subprocess launchers (Claude Agent SDK) to
    propagate the parent trace into a child process that emits its
    own OTel spans — the child's spans nest under the agent-plane
    root span in the same trace.

    :returns: A dict with ``TRACEPARENT`` (and optionally
        ``TRACESTATE``) suitable for merging into the ``env`` dict
        passed to ``subprocess.Popen`` or executor SDK options.
        Empty dict when no span is active.
    """
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    result: dict[str, str] = {}
    if "traceparent" in carrier:
        result["TRACEPARENT"] = carrier["traceparent"]
    if "tracestate" in carrier:
        result["TRACESTATE"] = carrier["tracestate"]
    return result


def init() -> None:
    """
    Initialize MLflow Tracing for the agent-plane runtime.

    Safe to call multiple times; the second and subsequent calls
    refresh the content-capture flag but do not re-register providers.

    Three modes based on the environment:

    * **OTLP export to an external collector.** When
      ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, we flip
      ``MLFLOW_ENABLE_OTLP_EXPORTER=true`` so MLflow exports via
      OTLP to the operator's collector (Jaeger, Tempo, MLflow
      tracking server's ``/v1/traces``, etc.) rather than to an
      MLflow tracking server's internal store.

    * **MLflow tracking server.** When ``OTEL_EXPORTER_OTLP_ENDPOINT``
      is unset but ``MLFLOW_TRACKING_URI`` is set, MLflow exports
      traces to the configured tracking server. We leave this path
      untouched.

    * **No-op.** When neither is set, MLflow emits no-op spans —
      zero overhead on span creation.

    Unified mode (``MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false``) is
    forced so the global OTel ``TracerProvider`` is shared between
    MLflow and raw OTel instrumentation (FastAPI, HTTPX).
    """
    global _capture_content, _initialized

    _capture_content = _env_bool("AGENT_PLANE_OTEL_CAPTURE_CONTENT")

    if _initialized:
        return

    # Unified provider mode: MLflow shares the global TracerProvider
    # with raw OTel instrumentation so FastAPI/HTTPX auto-instrumented
    # spans live in the same trace as our MLflow spans.
    os.environ.setdefault("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")

    # When an OTLP endpoint is configured, explicitly flip MLflow's
    # OTLP exporter flag. MLflow requires this in addition to the
    # standard OTel env vars (which it also respects).
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        os.environ.setdefault("MLFLOW_ENABLE_OTLP_EXPORTER", "true")

    try:
        import mlflow.tracing

        mlflow.tracing.enable()
    except Exception:
        _logger.exception("failed to initialize MLflow tracing")

    # NOTE: We do NOT install FastAPI / HTTPX auto-instrumentation
    # here. MLflow's span processor has a bug where it wraps raw
    # OTel spans (like those produced by auto-instrumentors) via
    # ``create_mlflow_span`` with a default ``span_type=None``,
    # which leaves ``mlflow.spanType`` serialized as the JSON
    # literal ``null``. When MLflow's OTel metrics mixin reads
    # this back to emit per-span metrics, ``try_json_loads``
    # returns ``None`` and the OTLP exporter crashes encoding
    # the metric attribute.
    #
    # Omitting FastAPI/HTTPX instrumentation means:
    #
    # * No automatic HTTP SERVER spans around request handlers.
    #   Operators rely on uvicorn access logs for HTTP-level
    #   observability instead.
    # * No automatic ``traceparent`` header injection on outbound
    #   ``httpx`` calls. The remote executor does not get
    #   distributed-tracing context propagation until this MLflow
    #   bug is fixed or we add the instrumentor in a way that
    #   excludes its spans from MLflow's processor.
    #
    # When the MLflow upstream fix lands, we can re-enable the
    # instrumentors here (guarded by ImportError fallthrough).

    _initialized = True
    _logger.info(
        "agent-plane telemetry initialized (endpoint=%s, capture_content=%s)",
        endpoint or "<none>",
        _capture_content,
    )
