"""Scoped, runtime policy stack for outbound ``AsyncHTTPClient`` requests.

The permanent knobs of the HTTP client (the implementation selected with
`AsyncHTTPClient.configure`, the ``max_clients`` concurrency cap, the
client's ``defaults`` and the per-request `HTTPRequest` fields) all describe
how the client behaves *forever*.  Sometimes a section of code -- a
middleware, a hook, a nested coroutine -- needs to temporarily change how
outbound requests are treated *within that section only*: refuse to start
any new fetches, fail immediately instead of queueing behind a full client,
or suspend timeout-induced failures while the section is in progress.

This module provides a policy stack for exactly that purpose.  Frames are
pushed with `push_outbound_policy` (or the `outbound_policy` context
manager) and popped again; the current frame set is bound to the running
cooperation scope with a `~contextvars.ContextVar`, so independent
coroutines and callbacks on the same `.IOLoop` never see each other's
frames.  A nested scope reads the stack as it exists *at the time of each
fetch*; it never keeps a snapshot taken when its parent scope was entered.

Each frame carries three independent gates:

* ``fetch_admit`` -- whether a new outbound fetch may be admitted at all
  (`FetchAdmit.HONOR_CFG`, `FetchAdmit.DENY_NEW`, `FetchAdmit.ONE_MORE`).
* ``timeout_gate`` -- how connect/request timeouts of fetches admitted in
  this frame are treated (`TimeoutGate.HONOR_CFG`, `TimeoutGate.FREEZE`,
  `TimeoutGate.SHRINK_NOW`).
* ``queue_gate`` -- what happens when the request would wait because the
  client is at capacity (`QueueGate.HONOR_CFG`, `QueueGate.FAIL_FAST`,
  `QueueGate.PASSTHROUGH`).

The stack is deliberately not stored on the `AsyncHTTPClient` or the
`.IOLoop`: it is orthogonal to the configured implementation, the
``max_clients`` cap and the client defaults, and a single frame governs
every client used from within the scope.
"""

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from tornado.httpclient import HTTPClientError

__all__ = [
    "FetchAdmit",
    "TimeoutGate",
    "QueueGate",
    "OutboundPolicyFrame",
    "OutboundPolicyError",
    "OutboundFetchDenied",
    "OutboundQueueFull",
    "OutboundPolicyConflict",
    "push_outbound_policy",
    "pop_outbound_policy",
    "outbound_policy",
    "current_outbound_policy",
]


class FetchAdmit(str, Enum):
    """Admission policy for new outbound fetches in a frame.

    * ``HONOR_CFG`` -- no admission gate beyond the configured client
      semantics (``max_clients`` queueing and the curl handle pool).
    * ``DENY_NEW`` -- every new outbound fetch attempted while the frame
      is active is rejected with an `OutboundFetchDenied` error.
      Requests already in flight are not affected.
    * ``ONE_MORE`` -- exactly one more outbound fetch is admitted by the
      frame; the next one is rejected.  The permit is consumed at
      admission time and is never refunded.
    """

    HONOR_CFG = "honor_cfg"
    DENY_NEW = "deny_new"
    ONE_MORE = "one_more"


class TimeoutGate(str, Enum):
    """How the connect/request timeouts of an admitted fetch are treated.

    * ``HONOR_CFG`` -- honor the effective ``connect_timeout`` /
      ``request_timeout`` (request fields merged with client defaults).
    * ``FREEZE`` -- while the frame is active, fetches admitted under it
      are not failed by a connect or whole-request timeout.  Real network
      errors still propagate normally and no successful empty response is
      fabricated; only the timeout failure surface is suspended.
    * ``SHRINK_NOW`` -- the effective waiting time is tightened to the
      frame's ``connect_timeout`` / ``request_timeout`` bounds (the
      smaller of the configured value and the bound).  Timeouts still
      surface through the ordinary HTTP client error path.
    """

    HONOR_CFG = "honor_cfg"
    FREEZE = "freeze"
    SHRINK_NOW = "shrink_now"


class QueueGate(str, Enum):
    """What happens when a fetch would wait for client capacity.

    * ``HONOR_CFG`` -- honor the configured queueing (Simple client queue
      or curl handle-pool waiting).
    * ``FAIL_FAST`` -- if the request would wait because ``max_clients``
      is saturated (or no curl handle is free), fail it immediately with
      an `OutboundQueueFull` error instead of taking a queue slot.
      Requests that can start immediately are unaffected.
    * ``PASSTHROUGH`` -- the frame does not alter queueing semantics.
    """

    HONOR_CFG = "honor_cfg"
    FAIL_FAST = "fail_fast"
    PASSTHROUGH = "passthrough"


class OutboundPolicyError(HTTPClientError):
    """Base class for failures raised by the outbound policy stack.

    Like other client-side failures these are reported as HTTP status 599.
    The ``gate`` attribute names the gate that produced the failure
    (``"deny_new"``, ``"one_more"``, ``"fail_fast"`` or ``"freeze"``).

    .. versionadded:: 6.7
    """

    def __init__(
        self,
        message: str,
        gate: Optional[str] = None,
        response: Any = None,
    ) -> None:
        self.gate = gate
        super().__init__(599, message=message, response=response)


class OutboundFetchDenied(OutboundPolicyError):
    """A new outbound fetch was rejected by a ``fetch_admit`` frame.

    Raised asynchronously on the `.Future` returned by
    `~tornado.httpclient.AsyncHTTPClient.fetch`; the request never reached
    the client implementation and holds no queue slot.

    .. versionadded:: 6.7
    """


class OutboundQueueFull(OutboundPolicyError):
    """A ``fail_fast`` frame found the client at capacity.

    The request would have waited in the ``max_clients`` queue (or for a
    free curl handle), so it was failed immediately without queueing.

    .. versionadded:: 6.7
    """


class OutboundPolicyConflict(OutboundPolicyError):
    """A frame could not be pushed because it conflicted with live state.

    Raised synchronously by `push_outbound_policy` when a ``freeze`` frame
    is pushed after a timeout governed by an outer frame has already
    started ticking: the running timer must be voided, recomputed or
    explicitly rejected, it may not silently keep using the frame that
    was current when the wait began.

    .. versionadded:: 6.7
    """


@dataclass
class OutboundPolicyFrame:
    """One frame of the outbound request policy stack.

    Instances are created by `push_outbound_policy` / `outbound_policy`;
    applications do not construct them directly.  The counter fields are
    mutable state shared by every task and callback that inherited the
    frame, which is what makes an ``one_more`` permit a property of the
    whole cooperation scope rather than of a single call.
    """

    fetch_admit: FetchAdmit = FetchAdmit.HONOR_CFG
    timeout_gate: TimeoutGate = TimeoutGate.HONOR_CFG
    queue_gate: QueueGate = QueueGate.HONOR_CFG
    connect_timeout: Optional[float] = None
    request_timeout: Optional[float] = None
    # Number of connect/request/queue timers currently ticking for
    # requests admitted under this frame.  Used to reject a ``freeze``
    # push that would otherwise silently fail to govern a started wait.
    armed_timeouts: int = field(default=0, init=False)
    # Whether the frame's single ``one_more`` permit has been consumed.
    one_more_used: bool = field(default=False, init=False)


# The live stack, bound to the current context (cooperation scope).  Each
# push replaces the tuple and returns a contextvars.Token, so pop is a
# simple reset and concurrent scopes on the same IOLoop cannot cross.
_stack: "contextvars.ContextVar[Optional[tuple[OutboundPolicyFrame, ...]]]" = (
    contextvars.ContextVar("tornado_outbound_policy_stack", default=None)
)


def _as_enum(value: Union[str, Enum], enum_cls: type, name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    raise ValueError(f"invalid {name}: {value!r}")


def push_outbound_policy(
    *,
    fetch_admit: Union[FetchAdmit, str] = FetchAdmit.HONOR_CFG,
    timeout_gate: Union[TimeoutGate, str] = TimeoutGate.HONOR_CFG,
    queue_gate: Union[QueueGate, str] = QueueGate.HONOR_CFG,
    connect_timeout: Optional[float] = None,
    request_timeout: Optional[float] = None,
) -> "contextvars.Token[Optional[tuple[OutboundPolicyFrame, ...]]]":
    """Pushes one frame onto the outbound request policy stack.

    The returned token must be passed to `pop_outbound_policy` (the
    `outbound_policy` context manager does this automatically, including
    when the body raises).

    Frames govern fetches *admitted* while they are active. A request
    already in flight when a frame is pushed is not cancelled or
    rewritten; a ``deny_new`` frame only blocks fetches attempted after
    the push, and a ``freeze`` frame only covers fetches admitted under
    it.

    A ``freeze`` frame cannot be pushed while a timeout of a request
    admitted under a frame still on the stack is already counting down;
    in that case `OutboundPolicyConflict` is raised and nothing is
    pushed (an already-started wait may not silently keep using the
    frame that was current when the wait began).

    ``connect_timeout`` / ``request_timeout`` bounds are only valid with
    ``timeout_gate="shrink_now"`` and must be positive; ``None`` leaves
    the corresponding timeout unchanged.
    """
    admit = _as_enum(fetch_admit, FetchAdmit, "fetch_admit")
    timeout = _as_enum(timeout_gate, TimeoutGate, "timeout_gate")
    queue = _as_enum(queue_gate, QueueGate, "queue_gate")
    for value, label in (
        (connect_timeout, "connect_timeout"),
        (request_timeout, "request_timeout"),
    ):
        if value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{label} must be a number, got {value!r}")
            if value <= 0:
                raise ValueError(
                    f"{label} must be a positive number of seconds, got {value!r}"
                )
    if timeout is not TimeoutGate.SHRINK_NOW:
        if connect_timeout is not None or request_timeout is not None:
            raise ValueError(
                "connect_timeout/request_timeout bounds require "
                'timeout_gate="shrink_now"'
            )
    elif connect_timeout is None and request_timeout is None:
        raise ValueError(
            'timeout_gate="shrink_now" requires at least one of '
            "connect_timeout/request_timeout"
        )

    stack = _stack.get()
    if timeout is TimeoutGate.FREEZE and stack:
        for frame in stack:
            if frame.armed_timeouts:
                raise OutboundPolicyConflict(
                    "cannot push a timeout_gate=freeze frame while a connect "
                    "or request timeout governed by an active policy frame "
                    "is already counting down; cancel or recompute the "
                    "running timeout before freezing",
                    gate=TimeoutGate.FREEZE.value,
                )

    frame = OutboundPolicyFrame(
        fetch_admit=admit,
        timeout_gate=timeout,
        queue_gate=queue,
        connect_timeout=float(connect_timeout) if connect_timeout is not None else None,
        request_timeout=float(request_timeout) if request_timeout is not None else None,
    )
    return _stack.set(tuple(stack) + (frame,) if stack else (frame,))


def pop_outbound_policy(
    token: "contextvars.Token[Optional[tuple[OutboundPolicyFrame, ...]]]",
) -> None:
    """Pops the frame identified by ``token`` from the stack.

    Must be called from the same context as the matching
    `push_outbound_policy`.  After popping, the outer frame (if any)
    governs again.
    """
    _stack.reset(token)


@contextlib.contextmanager
def outbound_policy(
    *,
    fetch_admit: Union[FetchAdmit, str] = FetchAdmit.HONOR_CFG,
    timeout_gate: Union[TimeoutGate, str] = TimeoutGate.HONOR_CFG,
    queue_gate: Union[QueueGate, str] = QueueGate.HONOR_CFG,
    connect_timeout: Optional[float] = None,
    request_timeout: Optional[float] = None,
) -> Iterator[OutboundPolicyFrame]:
    """Context manager that pushes a frame for the duration of its body.

    The frame is popped again when the body exits, including via
    exceptions::

        async def handler():
            with outbound_policy(fetch_admit="deny_new"):
                # No outbound fetches may start in this section.
                ...
    """
    token = push_outbound_policy(
        fetch_admit=fetch_admit,
        timeout_gate=timeout_gate,
        queue_gate=queue_gate,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )
    try:
        stack = _stack.get()
        assert stack is not None
        yield stack[-1]
    finally:
        _stack.reset(token)


def current_outbound_policy() -> Optional[tuple[OutboundPolicyFrame, ...]]:
    """Returns the policy frames active in the current context, innermost last.

    Returns ``None`` when no frame has been pushed, which is exactly the
    unmodified, configured-client behavior.
    """
    stack = _stack.get()
    return tuple(stack) if stack else None


class _PolicyView:
    """Resolved, per-fetch view of the stack at admission time.

    Carried on the internal request proxy so that the client
    implementations can consult it without modifying ``max_clients``,
    client defaults or `HTTPRequest` defaults.  Timeout values stay live
    through the proxy: implementations read ``request.connect_timeout`` /
    ``request.request_timeout`` as before and get the effective value
    (tightened or frozen) for free.
    """

    __slots__ = (
        "frame",
        "admit",
        "queue",
        "timeout",
        "connect_timeout",
        "request_timeout",
        "deny_gate",
    )

    def __init__(
        self,
        frame: OutboundPolicyFrame,
        admit: FetchAdmit,
        queue: QueueGate,
        timeout: TimeoutGate,
        connect_timeout_bound: Optional[float],
        request_timeout_bound: Optional[float],
        deny_gate: Optional[str],
    ) -> None:
        self.frame = frame
        self.admit = admit
        self.queue = queue
        self.timeout = timeout
        self.connect_timeout = connect_timeout_bound
        self.request_timeout = request_timeout_bound
        self.deny_gate = deny_gate

    def effective_timeout(self, name: str, configured: Optional[float]) -> float:
        """Returns the effective timeout for a proxied request attribute."""
        if self.timeout is TimeoutGate.FREEZE:
            # Both backends already treat 0 as "no timeout".
            return 0.0
        if self.timeout is TimeoutGate.SHRINK_NOW:
            bound = (
                self.connect_timeout
                if name == "connect_timeout"
                else self.request_timeout
            )
            if bound is None:
                return configured if configured is not None else 0.0
            # A configured 0 means "no timeout"; the tighter bound then
            # becomes the only limit instead of being swallowed by min().
            if not configured:
                return bound
            return min(configured, bound)
        return configured if configured is not None else 0.0


def _resolve(proxy: Any) -> Optional[_PolicyView]:
    """Resolves the live stack into a per-fetch view at fetch admission.

    ``proxy`` is the ``_RequestProxy`` (request fields already merged with
    client defaults).  Returns ``None`` when no frame is active, in which
    case every code path behaves exactly as it did without the stack.
    Consumes an ``one_more`` permit when one is granted.
    """
    stack = _stack.get()
    if not stack:
        return None

    admit = FetchAdmit.HONOR_CFG
    queue = QueueGate.HONOR_CFG
    timeout = TimeoutGate.HONOR_CFG
    connect_bound: Optional[float] = None
    request_bound: Optional[float] = None
    deny_gate: Optional[str] = None

    # Innermost non-honor frame on each axis decides; the others are
    # transparent to that axis.
    for frame in reversed(stack):
        if admit is FetchAdmit.HONOR_CFG and frame.fetch_admit is not FetchAdmit.HONOR_CFG:
            admit = frame.fetch_admit
        if (
            queue is QueueGate.HONOR_CFG
            and frame.queue_gate not in (QueueGate.HONOR_CFG, QueueGate.PASSTHROUGH)
        ):
            queue = frame.queue_gate
        if (
            timeout is TimeoutGate.HONOR_CFG
            and frame.timeout_gate is not TimeoutGate.HONOR_CFG
        ):
            timeout = frame.timeout_gate
            connect_bound = frame.connect_timeout
            request_bound = frame.request_timeout

    if admit is FetchAdmit.DENY_NEW:
        deny_gate = FetchAdmit.DENY_NEW.value
    elif admit is FetchAdmit.ONE_MORE:
        # The frame that supplied the one_more decision owns the permit.
        for frame in reversed(stack):
            if frame.fetch_admit is FetchAdmit.ONE_MORE:
                if frame.one_more_used:
                    deny_gate = FetchAdmit.ONE_MORE.value
                else:
                    frame.one_more_used = True
                break

    # The innermost frame at admission time owns this request's timers
    # for the purpose of late-freeze conflict detection.
    return _PolicyView(
        stack[-1],
        admit,
        queue,
        timeout,
        connect_bound,
        request_bound,
        deny_gate,
    )


def _fail_fast_active(view: Optional[_PolicyView]) -> bool:
    """True when a resolved view demands fail-fast queueing."""
    return view is not None and view.queue is QueueGate.FAIL_FAST


def _timeout_will_arm(view: Optional[_PolicyView], phase: str) -> None:
    """Records that a ``phase`` timeout started counting for ``view``."""
    if view is not None:
        view.frame.armed_timeouts += 1


def _timeout_did_disarm(view: Optional[_PolicyView], phase: str) -> None:
    """Records that a previously armed ``phase`` timeout no longer ticks."""
    if view is not None and view.frame.armed_timeouts > 0:
        view.frame.armed_timeouts -= 1
