#
# Copyright 2026 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""Inbound request-period policy stacks.

This module implements a *policy stack* scoped to a single inbound
``HTTPServer`` connection and to the request segment (or WebSocket
handshake/session) currently being processed on that connection.  It is
deliberately independent of the outbound HTTP client machinery
(`tornado.simple_httpclient`, `tornado.curl_httpclient`, connection
pooling) and of template/static-resource handling.

There is intentionally no process-wide singleton: each accepted TCP
connection gets its own `RequestPolicyStack`, so two requests that run
concurrently on two connections can never observe each other's frames.
Frames are pushed and popped like a stack; inner frames override outer
frames for the dimensions they specify, and popping an inner frame
restores the outer frame's view.

The four dimensions are independent enums (they must not collapse into
a single boolean):

* ``body_admit`` selects how the request body is delivered.
* ``write_fault`` selects how outbound write/error paths behave.
* ``upgrade_surface`` selects how WebSocket handshakes and early session
  exceptions are presented.
* ``conn_continue`` selects what happens to the HTTP/1 connection read
  loop after the response.

When no frame has ever been pushed, every read site observes the
``honor_*`` values and behavior is identical to the framework defaults.

Pushing a frame after the point where the corresponding decision had to
be made raises `PolicyPhaseError`; an earlier decision is never silently
replayed from a stale snapshot.
"""

import contextlib
import enum
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("tornado.inbound")

#: Attribute under which the per-connection stack is stored on the
#: opaque connection context object.
_STACK_ATTR = "_tornado_inbound_policy_stack"


class BodyAdmit(enum.Enum):
    """How the request body of the current segment may be delivered."""

    #: Keep the existing delivery: either ``request._parse_body`` for
    #: ordinary handlers or the ``stream_request_body`` path for
    #: decorated handlers.
    HONOR_BUFFER = "honor_buffer"

    #: Even if the handler class is not marked ``stream_request_body``,
    #: the segment must observably take the early ``execute()`` plus
    #: chunked ``data_received`` path.  The body must not be silently
    #: buffered as a whole and handled as an ordinary body.
    FORCE_STREAM = "force_stream"

    #: After the request headers, body bytes must not be accepted; any
    #: bytes that do arrive fail observably (HTTP 400 / disconnect) and
    #: must never be presented as a successful empty body.
    REJECT_BODY = "reject_body"


class WriteFault(enum.Enum):
    """How outbound writes and error responses behave in this segment."""

    #: ``write``/``flush``/``finish`` behave as usual.
    HONOR_FLUSH = "honor_flush"

    #: Every flush (or equivalent write) must fail observably; buffered
    #: content must not be reported as sent to the client.
    FAIL_FLUSH = "fail_flush"

    #: When ``send_error``/``write_error`` is triggered, no error page
    #: body is written; only the observable status code (and a log fork)
    #: remains.
    DROP_ON_ERROR = "drop_on_error"


class UpgradeSurface(enum.Enum):
    """How the WebSocket handshake and early session failures surface."""

    #: The WebSocket handshake/upgrade behaves as usual.
    HONOR_WS = "honor_ws"

    #: The WebSocket handshake fails observably through the ordinary
    #: HTTP error exit (``set_status`` + ``finish``); ``open`` is never
    #: entered.
    DENY_UPGRADE = "deny_upgrade"

    #: An uncaught exception in ``open`` or in the first ``on_message``
    #: observably closes the session with a WebSocket close frame; the
    #: connection must not linger half-open like HTTP keep-alive.
    CLOSE_ON_OPEN_EXC = "close_on_open_exc"


class ConnContinue(enum.Enum):
    """Whether the HTTP/1 server connection read loop continues."""

    #: Respect the existing keep-alive decision.
    HONOR_KEEPALIVE = "honor_keepalive"

    #: After this segment's response ends, the connection must not enter
    #: another iteration of the server request loop.
    FORCE_CLOSE = "force_close"

    #: If xheaders were applied through ``_ProxyAdapter`` for this
    #: segment, they must be strictly restored before the next request's
    #: headers, so protocol/remote-address views cannot leak.
    STICKY_XHEADERS = "sticky_xheaders"


class PolicyPhaseError(RuntimeError):
    """Raised when a frame is pushed after its decision point has passed.

    Tornado cannot retroactively change, for example, whether a body was
    streamed once ``headers_received`` started ``execute``, so the push
    is rejected instead of being silently ignored or snapshotted.
    """


class InboundFlushError(IOError):
    """Raised on the ``flush`` Future when ``fail_flush`` blocks a write.

    Nothing is written to the client when this error is reported and the
    underlying connection is closed so buffered output cannot be
    mistaken for delivered output.
    """


@dataclass(frozen=True)
class SegmentSummary:
    """Read-only bookkeeping left behind by one request segment."""

    body_decided: bool = False
    body_admit: Optional[BodyAdmit] = None
    writes_begun: bool = False
    ws_accept_begun: bool = False
    response_done: bool = False
    detached: bool = False
    force_close_committed: bool = False
    xheaders_applied: bool = False
    sticky_xheaders: bool = False
    xheaders_restored: bool = False
    leftover_frames_popped: int = 0


@dataclass(frozen=True)
class EffectivePolicy:
    """Read-only effective view for the current segment.

    The dimension fields are merged from the live stack frames
    (innermost frame wins per dimension) with one-time decisions latched
    on the segment, and combined with baseline state derived from
    ``Application.settings``, ``HTTP1ConnectionParameters`` and the
    current handler.
    """

    body_admit: BodyAdmit = BodyAdmit.HONOR_BUFFER
    write_fault: WriteFault = WriteFault.HONOR_FLUSH
    upgrade_surface: UpgradeSurface = UpgradeSurface.HONOR_WS
    conn_continue: ConnContinue = ConnContinue.HONOR_KEEPALIVE
    # Merged baseline state (not policy dimensions).
    stream_request_body: bool = False
    no_keep_alive: bool = False
    xheaders: bool = False
    # The live stack this view was computed from (None for defaults).
    stack: Optional["RequestPolicyStack"] = field(default=None, repr=False)


DEFAULT_EFFECTIVE = EffectivePolicy()


class _Segment:
    """Mutable phase/latch state for a single request segment."""

    def __init__(self) -> None:
        # Body delivery is decided once, in headers_received.
        self.body_decided = False
        self.latched_body_admit: Optional[BodyAdmit] = None
        # Becomes true once the handler has written anything.
        self.writes_begun = False
        # Becomes true once WebSocketProtocol.accept_connection starts.
        self.ws_accept_begun = False
        # Becomes true once the HTTP response is finished.
        self.response_done = False
        # Set once the stream has been detached (e.g. successful WS).
        self.detached = False
        # force_close that already took effect stays in force for the segment.
        self.force_close_committed = False
        # xheaders bookkeeping driven by _ProxyAdapter.
        self.xheaders_applied = False
        self.sticky_xheaders = False
        self.xheaders_restored = False

    def summary(self, leftover: int) -> SegmentSummary:
        return SegmentSummary(
            body_decided=self.body_decided,
            body_admit=self.latched_body_admit,
            writes_begun=self.writes_begun,
            ws_accept_begun=self.ws_accept_begun,
            response_done=self.response_done,
            detached=self.detached,
            force_close_committed=self.force_close_committed,
            xheaders_applied=self.xheaders_applied,
            sticky_xheaders=self.sticky_xheaders,
            xheaders_restored=self.xheaders_restored,
            leftover_frames_popped=leftover,
        )


class RequestPolicyFrame:
    """A single frame on a `RequestPolicyStack`.

    Instances are created by `RequestPolicyStack.push` (or
    `~RequestPolicyStack.frame`) and are opaque tokens; pass one to
    `RequestPolicyStack.pop` to remove exactly that frame.
    """

    __slots__ = (
        "body_admit",
        "write_fault",
        "upgrade_surface",
        "conn_continue",
        "tag",
    )

    def __init__(
        self,
        body_admit: BodyAdmit,
        write_fault: WriteFault,
        upgrade_surface: UpgradeSurface,
        conn_continue: ConnContinue,
        tag: Any,
    ) -> None:
        self.body_admit = body_admit
        self.write_fault = write_fault
        self.upgrade_surface = upgrade_surface
        self.conn_continue = conn_continue
        self.tag = tag

    def __repr__(self) -> str:
        return (
            "RequestPolicyFrame(body_admit=%s, write_fault=%s, "
            "upgrade_surface=%s, conn_continue=%s, tag=%r)"
            % (
                self.body_admit.value,
                self.write_fault.value,
                self.upgrade_surface.value,
                self.conn_continue.value,
                self.tag,
            )
        )


class RequestPolicyStack:
    """A push/pop policy stack bound to one inbound connection.

    Create one per accepted TCP connection with
    `bind_policy_stack` and read it from request/handler/connection
    objects with `policy_stack` / `effective_policy`.
    """

    def __init__(self) -> None:
        self._frames: list[RequestPolicyFrame] = []
        self._segment = _Segment()
        #: Summary of the previous segment, if any.
        self.last_summary: Optional[SegmentSummary] = None

    # ------------------------------------------------------------------
    # Segment lifecycle (driven by HTTP1ServerConnection).
    # ------------------------------------------------------------------

    def begin_segment(self) -> SegmentSummary:
        """Start a new request segment on this connection.

        Any frames left over from the previous segment are popped
        (exception paths can therefore be cleaned up even without a
        matching ``pop``), and the previous segment's summary is
        returned.
        """
        leftover = len(self._frames)
        if leftover:
            # Debug level: this is a deliberate safety net for exception
            # paths that could not pop their frames; the new segment must
            # always start with a clean stack regardless.
            logger.debug(
                "inbound policy stack: %d frame(s) left over from the previous "
                "request segment were popped",
                leftover,
            )
            summary = self._segment.summary(leftover)
            self._frames.clear()
        else:
            summary = self._segment.summary(0)
        self.last_summary = summary
        self._segment = _Segment()
        return summary

    @property
    def segment(self) -> _Segment:
        """The current, mutable per-segment phase/latch state."""
        return self._segment

    # ------------------------------------------------------------------
    # Push / pop.
    # ------------------------------------------------------------------

    def push(
        self,
        *,
        body_admit: BodyAdmit = BodyAdmit.HONOR_BUFFER,
        write_fault: WriteFault = WriteFault.HONOR_FLUSH,
        upgrade_surface: UpgradeSurface = UpgradeSurface.HONOR_WS,
        conn_continue: ConnContinue = ConnContinue.HONOR_KEEPALIVE,
        tag: Any = None,
    ) -> RequestPolicyFrame:
        """Push a frame onto the stack and return its opaque token.

        Raises `PolicyPhaseError` when a non-default value arrives too
        late for the decision it would change.
        """
        if not isinstance(body_admit, BodyAdmit):
            raise TypeError("body_admit must be a BodyAdmit member")
        if not isinstance(write_fault, WriteFault):
            raise TypeError("write_fault must be a WriteFault member")
        if not isinstance(upgrade_surface, UpgradeSurface):
            raise TypeError("upgrade_surface must be an UpgradeSurface member")
        if not isinstance(conn_continue, ConnContinue):
            raise TypeError("conn_continue must be a ConnContinue member")

        seg = self._segment
        if body_admit is not BodyAdmit.HONOR_BUFFER and seg.body_decided:
            raise PolicyPhaseError(
                "cannot set body_admit=%s after headers_received started execute; "
                "body delivery for this segment was already decided (%s)"
                % (body_admit.value, seg.latched_body_admit)
            )
        if write_fault is WriteFault.FAIL_FLUSH and seg.writes_begun:
            raise PolicyPhaseError(
                "cannot set write_fault=fail_flush after write()/flush() has "
                "already happened in this segment"
            )
        if upgrade_surface is UpgradeSurface.DENY_UPGRADE and seg.ws_accept_begun:
            raise PolicyPhaseError(
                "cannot set upgrade_surface=deny_upgrade after the WebSocket "
                "handshake entered accept_connection"
            )
        if conn_continue is ConnContinue.FORCE_CLOSE and seg.response_done:
            raise PolicyPhaseError(
                "cannot set conn_continue=force_close after the response was "
                "already finished"
            )

        frame = RequestPolicyFrame(
            body_admit, write_fault, upgrade_surface, conn_continue, tag
        )
        self._frames.append(frame)
        return frame

    def pop(self, frame: RequestPolicyFrame) -> None:
        """Remove exactly ``frame`` (no error if it is no longer present).

        Decisions already latched for the segment (e.g. body delivery or
        a committed force-close) stay in effect for the rest of the
        segment; they cannot be un-made by popping the frame.
        """
        try:
            self._frames.remove(frame)
        except ValueError:
            pass

    @contextlib.contextmanager
    def frame(
        self,
        *,
        body_admit: BodyAdmit = BodyAdmit.HONOR_BUFFER,
        write_fault: WriteFault = WriteFault.HONOR_FLUSH,
        upgrade_surface: UpgradeSurface = UpgradeSurface.HONOR_WS,
        conn_continue: ConnContinue = ConnContinue.HONOR_KEEPALIVE,
        tag: Any = None,
    ) -> Iterator[RequestPolicyFrame]:
        """Context manager that pushes a frame and always pops it again.

        This is the intended way to scope a policy to a middleware block
        or nested call; exception paths are cleaned up automatically::

            with stack.frame(write_fault=WriteFault.FAIL_FLUSH):
                ...
        """
        pushed = self.push(
            body_admit=body_admit,
            write_fault=write_fault,
            upgrade_surface=upgrade_surface,
            conn_continue=conn_continue,
            tag=tag,
        )
        try:
            yield pushed
        finally:
            self.pop(pushed)

    # ------------------------------------------------------------------
    # Effective view.
    # ------------------------------------------------------------------

    def effective(
        self,
        settings: Optional[dict[str, Any]] = None,
        params: Any = None,
        handler: Any = None,
    ) -> EffectivePolicy:
        """Compute the live read-only effective view.

        Frames are scanned innermost-first; each dimension takes the
        first non-default value encountered. Segment latches override
        frames for decisions that can only be made once.
        """
        seg = self._segment
        body_admit = seg.latched_body_admit
        write_fault: Optional[WriteFault] = None
        upgrade_surface: Optional[UpgradeSurface] = None
        conn_continue: Optional[ConnContinue] = None

        for frame in reversed(self._frames):
            if body_admit is None and frame.body_admit is not BodyAdmit.HONOR_BUFFER:
                body_admit = frame.body_admit
            if write_fault is None and frame.write_fault is not WriteFault.HONOR_FLUSH:
                write_fault = frame.write_fault
            if (
                upgrade_surface is None
                and frame.upgrade_surface is not UpgradeSurface.HONOR_WS
            ):
                upgrade_surface = frame.upgrade_surface
            if (
                conn_continue is None
                and frame.conn_continue is not ConnContinue.HONOR_KEEPALIVE
            ):
                conn_continue = frame.conn_continue

        if seg.sticky_xheaders and conn_continue is None:
            conn_continue = ConnContinue.STICKY_XHEADERS
        if seg.force_close_committed:
            conn_continue = ConnContinue.FORCE_CLOSE

        stream_request_body = bool(
            getattr(type(handler), "_stream_request_body", False)
        )
        no_keep_alive = bool(getattr(params, "no_keep_alive", False))
        xheaders = False
        if settings:
            xheaders = bool(settings.get("xheaders", False))

        return EffectivePolicy(
            body_admit=body_admit or BodyAdmit.HONOR_BUFFER,
            write_fault=write_fault or WriteFault.HONOR_FLUSH,
            upgrade_surface=upgrade_surface or UpgradeSurface.HONOR_WS,
            conn_continue=conn_continue or ConnContinue.HONOR_KEEPALIVE,
            stream_request_body=stream_request_body,
            no_keep_alive=no_keep_alive,
            xheaders=xheaders,
            stack=self,
        )

    # ------------------------------------------------------------------
    # Phase marks / latches (called by the framework's decision sites).
    # ------------------------------------------------------------------

    def mark_body_decision(self, value: BodyAdmit) -> None:
        """Latch the body delivery decision taken at ``headers_received``."""
        seg = self._segment
        seg.body_decided = True
        seg.latched_body_admit = value

    def mark_writes_begun(self) -> None:
        self._segment.writes_begun = True

    def mark_ws_accept_begun(self) -> None:
        self._segment.ws_accept_begun = True

    def mark_response_done(self) -> None:
        self._segment.response_done = True

    def note_detach(self) -> None:
        self._segment.detached = True

    def commit_force_close(self) -> None:
        """Remember that this connection will not serve another request."""
        self._segment.force_close_committed = True

    def note_xheaders_applied(self, sticky: bool) -> None:
        seg = self._segment
        seg.xheaders_applied = True
        if sticky:
            seg.sticky_xheaders = True

    def note_xheaders_restored(self) -> None:
        self._segment.xheaders_restored = True

    def should_end_connection(self) -> bool:
        """Read by the server request loop after a response completes."""
        seg = self._segment
        if seg.force_close_committed:
            return True
        return self.effective().conn_continue is ConnContinue.FORCE_CLOSE


# ----------------------------------------------------------------------
# Binding / resolution helpers.
# ----------------------------------------------------------------------


def bind_policy_stack(
    context: Any, stack: Optional[RequestPolicyStack] = None
) -> RequestPolicyStack:
    """Bind a fresh `RequestPolicyStack` to a connection context object."""
    if stack is None:
        stack = RequestPolicyStack()
    setattr(context, _STACK_ATTR, stack)
    return stack


def policy_stack(obj: Any) -> Optional[RequestPolicyStack]:
    """Return the stack bound to ``obj`` or ``None``.

    ``obj`` may be a connection context, an ``HTTPConnection``,
    an `.HTTPServerRequest`, a `~tornado.web.RequestHandler` or a
    WebSocket protocol/handler object that references one of those.
    """
    if obj is None:
        return None
    # The object itself may carry the stack (connection context).
    stack = getattr(obj, _STACK_ATTR, None)
    if stack is not None:
        return stack
    # HTTP1Connection / HTTP1ServerConnection expose ``context``.
    context = getattr(obj, "context", None)
    if context is not None and context is not obj:
        stack = getattr(context, _STACK_ATTR, None)
        if stack is not None:
            return stack
    # HTTPServerRequest exposes ``connection``.
    connection = getattr(obj, "connection", None)
    if connection is not None and connection is not obj:
        stack = policy_stack(connection)
        if stack is not None:
            return stack
    # RequestHandler / WebSocketProtocol expose ``request``.
    request = getattr(obj, "request", None)
    if request is not None and request is not obj:
        stack = policy_stack(request)
        if stack is not None:
            return stack
    return None


def effective_policy(obj: Any) -> EffectivePolicy:
    """Return the live effective policy view for ``obj``.

    Returns a default ``honor_*`` view (never ``None``) when no stack is
    bound, so read sites behave exactly like the existing APIs when no
    frame has been pushed.
    """
    stack = policy_stack(obj)
    if stack is None:
        return _baseline_for(obj)

    settings: Optional[dict[str, Any]] = None
    params: Any = None
    handler: Any = None

    request = getattr(obj, "request", None)
    if request is not None and getattr(request, "connection", None) is not None:
        # ``obj`` looks like a RequestHandler.
        handler = obj
        application = getattr(obj, "application", None)
        settings = getattr(application, "settings", None)
        params = getattr(request.connection, "params", None)
    else:
        connection = getattr(obj, "connection", None)
        if connection is not None:
            # ``obj`` looks like an HTTPServerRequest.
            request = obj
            params = getattr(connection, "params", None)
        elif getattr(obj, "params", None) is not None:
            # ``obj`` looks like an HTTP1Connection.
            params = obj.params

    return stack.effective(settings=settings, params=params, handler=handler)


def _baseline_for(obj: Any) -> EffectivePolicy:
    request = getattr(obj, "request", None)
    stream_request_body = bool(getattr(type(obj), "_stream_request_body", False))
    no_keep_alive = False
    xheaders = False
    if request is not None:
        application = getattr(obj, "application", None)
        settings = getattr(application, "settings", None)
        if settings:
            xheaders = bool(settings.get("xheaders", False))
        connection = getattr(request, "connection", None)
        params = getattr(connection, "params", None)
        no_keep_alive = bool(getattr(params, "no_keep_alive", False))
    elif getattr(obj, "params", None) is not None:
        no_keep_alive = bool(getattr(obj.params, "no_keep_alive", False))
    return EffectivePolicy(
        stream_request_body=stream_request_body,
        no_keep_alive=no_keep_alive,
        xheaders=xheaders,
    )
