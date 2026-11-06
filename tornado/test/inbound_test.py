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

"""Tests for the inbound request-period policy stack (`tornado.inbound`).

The tests assert observable outcomes on the wire (or on the WebSocket
session), push-timing rejection, and per-connection concurrency
isolation for each of the four policy dimensions.
"""

import asyncio
import contextlib
import json
import logging
import socket

from tornado import gen
from tornado.escape import json_decode
from tornado.httpclient import HTTPError
from tornado.httpserver import HTTPServer, _ProxyAdapter
from tornado.httputil import HTTPHeaders, HTTPMessageDelegate
from tornado import inbound
from tornado.iostream import IOStream
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.testing import ExpectLog, gen_test
from tornado.test.util import AsyncHTTPTestCase, TestCase
from tornado.web import Application, RequestHandler, stream_request_body
from tornado.websocket import WebSocketHandler, websocket_connect

# ---------------------------------------------------------------------------
# Connection-level middleware: pushes a frame around the whole message
# delegate (outside _ProxyAdapter and _HandlerDelegate) so the frame is
# live before headers_received makes the body-delivery decision.
# ---------------------------------------------------------------------------


class _FrameMessageDelegate(HTTPMessageDelegate):
    def __init__(self, delegate, stack, frame_kwargs):
        self._delegate = delegate
        self._stack = stack
        self._frame_kwargs = frame_kwargs
        self._cm = None

    def headers_received(self, start_line, headers):
        self._cm = self._stack.frame(**self._frame_kwargs)
        self._cm.__enter__()
        try:
            return self._delegate.headers_received(start_line, headers)
        except BaseException:
            self._pop()
            raise

    def data_received(self, chunk):
        return self._delegate.data_received(chunk)

    def finish(self):
        try:
            return self._delegate.finish()
        finally:
            self._pop()

    def on_connection_close(self):
        try:
            self._delegate.on_connection_close()
        finally:
            self._pop()

    def _pop(self):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None


class _FrameHTTPServer(HTTPServer):
    #: Keyword arguments for RequestPolicyStack.frame for every request.
    frame_kwargs = None

    def start_request(self, server_conn, request_conn):
        delegate = super().start_request(server_conn, request_conn)
        if self.frame_kwargs is not None:
            stack = inbound.policy_stack(request_conn)
            assert stack is not None
            delegate = _FrameMessageDelegate(delegate, stack, dict(self.frame_kwargs))
        return delegate


# ---------------------------------------------------------------------------
# Pure stack semantics
# ---------------------------------------------------------------------------


class PolicyStackUnitTest(TestCase):
    def test_default_view_is_all_honor(self):
        eff = inbound.EffectivePolicy()
        self.assertIs(eff.body_admit, inbound.BodyAdmit.HONOR_BUFFER)
        self.assertIs(eff.write_fault, inbound.WriteFault.HONOR_FLUSH)
        self.assertIs(eff.upgrade_surface, inbound.UpgradeSurface.HONOR_WS)
        self.assertIs(eff.conn_continue, inbound.ConnContinue.HONOR_KEEPALIVE)

    def test_no_stack_resolves_to_default(self):
        eff = inbound.effective_policy(object())
        self.assertIs(eff.body_admit, inbound.BodyAdmit.HONOR_BUFFER)
        self.assertIs(eff.write_fault, inbound.WriteFault.HONOR_FLUSH)
        self.assertIs(eff.stack, None)

    def test_push_pop_precedence_and_outer_frame(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        outer = stack.push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        self.assertIs(stack.effective().write_fault, inbound.WriteFault.FAIL_FLUSH)
        inner = stack.push(write_fault=inbound.WriteFault.DROP_ON_ERROR)
        self.assertIs(stack.effective().write_fault, inbound.WriteFault.DROP_ON_ERROR)
        stack.pop(inner)
        self.assertIs(stack.effective().write_fault, inbound.WriteFault.FAIL_FLUSH)
        stack.pop(outer)
        self.assertIs(stack.effective().write_fault, inbound.WriteFault.HONOR_FLUSH)

    def test_frame_context_manager_pops_on_exception(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        with self.assertRaises(ZeroDivisionError):
            with stack.frame(conn_continue=inbound.ConnContinue.FORCE_CLOSE):
                self.assertIs(
                    stack.effective().conn_continue,
                    inbound.ConnContinue.FORCE_CLOSE,
                )
                1 / 0
        self.assertIs(
            stack.effective().conn_continue,
            inbound.ConnContinue.HONOR_KEEPALIVE,
        )

    def test_phase_guard_body(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        stack.mark_body_decision(inbound.BodyAdmit.HONOR_BUFFER)
        with self.assertRaises(inbound.PolicyPhaseError):
            stack.push(body_admit=inbound.BodyAdmit.FORCE_STREAM)
        with self.assertRaises(inbound.PolicyPhaseError):
            stack.push(body_admit=inbound.BodyAdmit.REJECT_BODY)
        # honor remains a legal no-op after the decision.
        stack.push(body_admit=inbound.BodyAdmit.HONOR_BUFFER)

    def test_phase_guard_write(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        stack.mark_writes_begun()
        with self.assertRaises(inbound.PolicyPhaseError):
            stack.push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        # other write_fault values are not about the pre-write decision.
        stack.push(write_fault=inbound.WriteFault.DROP_ON_ERROR)

    def test_phase_guard_ws(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        stack.mark_ws_accept_begun()
        with self.assertRaises(inbound.PolicyPhaseError):
            stack.push(upgrade_surface=inbound.UpgradeSurface.DENY_UPGRADE)
        # close_on_open_exc governs open()/on_message() which follow.
        stack.push(upgrade_surface=inbound.UpgradeSurface.CLOSE_ON_OPEN_EXC)

    def test_phase_guard_response_done(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        stack.mark_response_done()
        with self.assertRaises(inbound.PolicyPhaseError):
            stack.push(conn_continue=inbound.ConnContinue.FORCE_CLOSE)
        stack.push(conn_continue=inbound.ConnContinue.STICKY_XHEADERS)

    def test_decision_latch_survives_pop(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        frame = stack.push(body_admit=inbound.BodyAdmit.FORCE_STREAM)
        stack.mark_body_decision(stack.effective().body_admit)
        stack.pop(frame)
        # The headers decision cannot be un-made by popping.
        self.assertIs(stack.effective().body_admit, inbound.BodyAdmit.FORCE_STREAM)

    def test_begin_segment_pops_leftover_frames(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        stack.push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        with ExpectLog(
            "tornado.inbound",
            ".*left over",
            required=True,
            level=logging.DEBUG,
        ):
            summary = stack.begin_segment()
        self.assertEqual(summary.leftover_frames_popped, 1)
        self.assertIs(stack.effective().write_fault, inbound.WriteFault.HONOR_FLUSH)

    def test_two_stacks_are_independent(self):
        first = inbound.RequestPolicyStack()
        second = inbound.RequestPolicyStack()
        first.begin_segment()
        second.begin_segment()
        first.push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        second.push(write_fault=inbound.WriteFault.DROP_ON_ERROR)
        self.assertIs(first.effective().write_fault, inbound.WriteFault.FAIL_FLUSH)
        self.assertIs(second.effective().write_fault, inbound.WriteFault.DROP_ON_ERROR)
        self.assertIsNot(first.segment, second.segment)

    def test_bad_enum_values_rejected(self):
        stack = inbound.RequestPolicyStack()
        stack.begin_segment()
        with self.assertRaises(TypeError):
            stack.push(body_admit="force_stream")
        with self.assertRaises(TypeError):
            stack.push(write_fault=True)


# ---------------------------------------------------------------------------
# Shared handlers
# ---------------------------------------------------------------------------


class _BodyRecorder:
    def __init__(self):
        self.events = []
        self.chunks = []
        self.body_attr = None
        self.flush_raised = False
        self.phase_rejected = False
        self.default_view = None


class BufferedControlHandler(RequestHandler):
    def initialize(self):
        self.rec = self.settings["rec"]

    def prepare(self):
        self.rec.events.append(("prepare", None))

    def post(self):
        self.rec.events.append(("post", self.request.body))
        self.rec.body_attr = self.request.body
        self.finish(b"buffered:" + self.request.body)


@stream_request_body
class StreamedControlHandler(RequestHandler):
    def initialize(self):
        self.rec = self.settings["rec"]

    def prepare(self):
        self.rec.events.append(("prepare", None))

    def data_received(self, data):
        self.rec.chunks.append(data)
        self.rec.events.append(("data", data))

    def post(self):
        self.rec.events.append(("post", None))
        self.finish(b"streamed:" + b"|".join(self.rec.chunks))


class ForcedStreamHandler(RequestHandler):
    # NOTE: deliberately NOT decorated with @stream_request_body.
    def initialize(self):
        self.rec = self.settings["rec"]

    def prepare(self):
        self.rec.events.append(("prepare", None))

    def data_received(self, data):
        self.rec.chunks.append(data)
        self.rec.events.append(("data", data))

    def post(self):
        self.rec.events.append(("post", None))
        self.rec.body_attr = self.request.body
        self.finish(b"forced:" + b"|".join(self.rec.chunks))


class RejectBodyHandler(RequestHandler):
    def initialize(self):
        self.settings["rec"].events.append(("initialize", None))

    def post(self):
        self.settings["rec"].events.append(("post", None))
        self.finish(b"should-not-happen-with-bytes")


class LateForceStreamHandler(RequestHandler):
    def prepare(self):
        try:
            inbound.policy_stack(self).push(body_admit=inbound.BodyAdmit.FORCE_STREAM)
        except inbound.PolicyPhaseError:
            self.settings["rec"].phase_rejected = True

    def post(self):
        self.finish(b"late-ok:" + self.request.body)


class FlushFailHandler(RequestHandler):
    async def get(self):
        rec = self.settings["rec"]
        inbound.policy_stack(self).push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        self.write(b"buffered-but-never-sent")
        future = self.flush()
        try:
            await future
        except inbound.InboundFlushError:
            rec.flush_raised = True
        # The connection was closed by the policy; do not call finish().


class LateFailFlushHandler(RequestHandler):
    def get(self):
        rec = self.settings["rec"]
        self.write(b"first")
        try:
            inbound.policy_stack(self).push(write_fault=inbound.WriteFault.FAIL_FLUSH)
        except inbound.PolicyPhaseError:
            rec.phase_rejected = True
        self.finish(b"late-write-ok")


class BoomHandler(RequestHandler):
    def prepare(self):
        if self.settings.get("drop"):
            inbound.policy_stack(self).push(
                write_fault=inbound.WriteFault.DROP_ON_ERROR
            )

    def get(self):
        raise ValueError("boom")


class SendErrorDropHandler(RequestHandler):
    def prepare(self):
        inbound.policy_stack(self).push(write_fault=inbound.WriteFault.DROP_ON_ERROR)

    def get(self):
        self.send_error(503)


class DefaultViewHandler(RequestHandler):
    def get(self):
        eff = self.inbound_effective
        assert eff.body_admit is inbound.BodyAdmit.HONOR_BUFFER
        assert eff.write_fault is inbound.WriteFault.HONOR_FLUSH
        assert eff.upgrade_surface is inbound.UpgradeSurface.HONOR_WS
        assert eff.conn_continue is inbound.ConnContinue.HONOR_KEEPALIVE
        self.settings["rec"].default_view = eff
        self.finish(b"default-ok")


class ForceCloseHandler(RequestHandler):
    def get(self):
        inbound.policy_stack(self).push(conn_continue=inbound.ConnContinue.FORCE_CLOSE)
        self.finish(b"closing")


class LateForceCloseHandler(RequestHandler):
    def get(self):
        self.finish(b"late-force-close-ok")

    def on_finish(self):
        try:
            inbound.policy_stack(self).push(
                conn_continue=inbound.ConnContinue.FORCE_CLOSE
            )
        except inbound.PolicyPhaseError:
            self.settings["rec"].phase_rejected = True


class XProtoHandler(RequestHandler):
    def initialize(self):
        self.rec = self.settings["rec"]

    def get(self):
        self.finish(json.dumps({"protocol": self.request.protocol}).encode())

    def on_finish(self):
        stack = inbound.policy_stack(self)
        if stack is not None:
            self.rec.xheaders_restored.append(stack.segment.xheaders_restored)


# ---------------------------------------------------------------------------
# body_admit
# ---------------------------------------------------------------------------


class BodyAdmitControlTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = _BodyRecorder()
        return Application(
            [
                ("/buffered", BufferedControlHandler),
                ("/streamed", StreamedControlHandler),
                ("/late", LateForceStreamHandler),
            ],
            rec=self.rec,
        )

    def test_late_force_stream_push_is_rejected(self):
        response = self.fetch(
            "/late",
            method="POST",
            body=b"xyz",
            allow_nonstandard_methods=True,
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"late-ok:xyz")
        self.assertTrue(self.rec.phase_rejected)

    def test_ordinary_post_buffers_whole_body(self):
        response = self.fetch(
            "/buffered",
            method="POST",
            body=b"abcdef",
            allow_nonstandard_methods=True,
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"buffered:abcdef")
        self.assertEqual(self.rec.body_attr, b"abcdef")
        # No data_received in buffered mode; prepare runs after the body.
        self.assertEqual(
            self.rec.events,
            [("prepare", None), ("post", b"abcdef")],
        )

    def test_stream_request_body_decorator_path(self):
        response = self.fetch(
            "/streamed",
            method="POST",
            body=b"abcdef",
            allow_nonstandard_methods=True,
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"streamed:abcdef")
        self.assertEqual(self.rec.events[0], ("prepare", None))
        self.assertTrue(all(kind == "data" for kind, _ in self.rec.events[1:-1]))
        self.assertEqual(self.rec.events[-1], ("post", None))


class _ForceStreamServer(_FrameHTTPServer):
    frame_kwargs = {"body_admit": inbound.BodyAdmit.FORCE_STREAM}


class BodyAdmitForceStreamTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = _BodyRecorder()
        return Application([("/force", ForcedStreamHandler)], rec=self.rec)

    def get_http_server(self):
        return _ForceStreamServer(self._app)

    def _connect(self):
        stream = IOStream(socket.socket())
        return stream, gen.convert_yielded(
            stream.connect(("127.0.0.1", self.get_http_port()))
        )

    @gen_test
    def test_undecorated_handler_takes_streaming_path(self):
        stream, connect_fut = self._connect()
        yield connect_fut
        try:
            stream.write(
                b"POST /force HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
                b"3\r\nabc\r\n"
            )
            # Let the first chunk be delivered before sending the next.
            yield asyncio.sleep(0.1)
            stream.write(b"3\r\ndef\r\n0\r\n\r\n")

            status = yield stream.read_until(b"\r\n")
            self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)
            header_bytes = yield stream.read_until(b"\r\n\r\n")
            headers = HTTPHeaders.parse(header_bytes.decode("latin1"))
            body = yield stream.read_bytes(int(headers["Content-Length"]))
            self.assertEqual(body, b"forced:abc|def")
        finally:
            stream.close()

        # Early execute + per-chunk data_received, observably ordered.
        kinds = [kind for kind, _ in self.rec.events]
        self.assertEqual(kinds[0], "prepare")
        self.assertLess(kinds.index("prepare"), kinds.index("data"))
        self.assertLess(kinds.index("data"), kinds.index("post"))
        self.assertEqual(kinds.count("data"), 2)
        self.assertEqual(self.rec.chunks, [b"abc", b"def"])
        # A streamed body is not exposed as request.body.
        self.assertEqual(self.rec.body_attr, b"")

    def test_force_stream_get_with_empty_body(self):
        response = self.fetch("/force")
        # Handler only defines post; empty body still takes normal flow.
        self.assertEqual(response.code, 405)


class _RejectBodyServer(_FrameHTTPServer):
    frame_kwargs = {"body_admit": inbound.BodyAdmit.REJECT_BODY}


class BodyAdmitRejectTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = _BodyRecorder()
        return Application([("/reject", RejectBodyHandler)], rec=self.rec)

    def get_http_server(self):
        return _RejectBodyServer(self._app)

    @gen_test
    def test_body_bytes_fail_observably_and_disconnect(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        try:
            with ExpectLog(gen_log, "Malformed HTTP message", level=logging.INFO):
                stream.write(
                    b"POST /reject HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Length: 5\r\n\r\n"
                    b"hello"
                )
                data = yield stream.read_until_close()
            self.assertTrue(
                data.startswith(b"HTTP/1.1 400"),
                f"expected 400 failure, got {data!r}",
            )
        finally:
            stream.close()
        # The handler never ran; no empty-body success was fabricated.
        self.assertEqual(self.rec.events, [])

    def test_zero_length_body_is_not_rejected(self):
        response = self.fetch(
            "/reject",
            method="POST",
            body=b"",
            allow_nonstandard_methods=True,
        )
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"should-not-happen-with-bytes")


# ---------------------------------------------------------------------------
# write_fault
# ---------------------------------------------------------------------------


class WriteFaultTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = _BodyRecorder()
        return Application(
            [
                ("/fail_flush", FlushFailHandler),
                ("/late_flush", LateFailFlushHandler),
                ("/boom", BoomHandler),
                ("/send_err", SendErrorDropHandler),
                ("/default", DefaultViewHandler),
            ],
            rec=self.rec,
        )

    @gen_test
    def test_fail_flush_sends_nothing_and_closes(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        with contextlib.closing(stream):
            with ExpectLog(gen_log, ".*fail_flush"):
                stream.write(b"GET /fail_flush HTTP/1.1\r\nHost: x\r\n\r\n")
                data = yield stream.read_until_close()
            self.assertEqual(data, b"")
        self.assertTrue(self.rec.flush_raised)

    def test_late_fail_flush_push_is_rejected(self):
        response = self.fetch("/late_flush")
        self.assertEqual(response.code, 200)
        # The already-buffered chunk is delivered normally after the
        # rejected push; no failure is fabricated.
        self.assertEqual(response.body, b"firstlate-write-ok")
        self.assertTrue(self.rec.phase_rejected)

    def test_drop_on_error_keeps_status_without_body(self):
        self._app.settings["drop"] = True
        with ExpectLog(app_log, "Uncaught exception"):
            with ExpectLog(gen_log, ".*drop_on_error"):
                response = self.fetch("/boom")
        self.assertEqual(response.code, 500)
        self.assertEqual(response.body, b"")
        self.assertEqual(response.headers.get("Content-Length"), "0")

    def test_control_error_path_still_writes_page(self):
        self._app.settings["drop"] = False
        with ExpectLog(app_log, "Uncaught exception", required=False):
            response = self.fetch("/boom")
        self.assertEqual(response.code, 500)
        self.assertIn(b"500", response.body)

    def test_drop_on_explicit_send_error(self):
        with ExpectLog(gen_log, ".*drop_on_error"):
            response = self.fetch("/send_err")
        self.assertEqual(response.code, 503)
        self.assertEqual(response.body, b"")
        self.assertEqual(response.headers.get("Content-Length"), "0")

    def test_empty_stack_view_equals_defaults(self):
        response = self.fetch("/default")
        self.assertEqual(response.code, 200)
        self.assertIsNotNone(self.rec.default_view)


# ---------------------------------------------------------------------------
# upgrade_surface
# ---------------------------------------------------------------------------


class DenyUpgradeWS(WebSocketHandler):
    def initialize(self):
        self.rec = self.settings["rec"]

    def prepare(self):
        inbound.policy_stack(self).push(
            upgrade_surface=inbound.UpgradeSurface.DENY_UPGRADE
        )

    def open(self):
        self.rec["ws_open_called"] = True

    def on_message(self, message):
        pass


class EchoWS(WebSocketHandler):
    def on_message(self, message):
        self.write_message(message)


class OpenBoomWS(WebSocketHandler):
    def initialize(self, close=None):
        if close is not None:
            inbound.policy_stack(self).push(
                upgrade_surface=inbound.UpgradeSurface.CLOSE_ON_OPEN_EXC
            )

    def open(self):
        raise RuntimeError("boom in open")

    def on_message(self, message):
        pass


class MessageBoomWS(WebSocketHandler):
    def initialize(self, close=None):
        if close is not None:
            inbound.policy_stack(self).push(
                upgrade_surface=inbound.UpgradeSurface.CLOSE_ON_OPEN_EXC
            )

    def open(self):
        pass

    def on_message(self, message):
        raise RuntimeError("boom in on_message")


class LateDenyWS(WebSocketHandler):
    def open(self):
        # Too late: accept_connection is already underway.
        try:
            inbound.policy_stack(self).push(
                upgrade_surface=inbound.UpgradeSurface.DENY_UPGRADE
            )
        except inbound.PolicyPhaseError:
            self.settings["rec"]["deny_phase_rejected"] = True

    def on_message(self, message):
        self.write_message(message)


class UpgradeSurfaceTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = {
            "ws_open_called": False,
            "deny_phase_rejected": False,
        }
        return Application(
            [
                ("/deny", DenyUpgradeWS),
                ("/echo", EchoWS),
                ("/open_boom", OpenBoomWS, dict(close=True)),
                ("/open_boom_honor", OpenBoomWS),
                ("/msg_boom", MessageBoomWS, dict(close=True)),
                ("/msg_boom_honor", MessageBoomWS),
                ("/late_deny", LateDenyWS),
            ],
            rec=self.rec,
        )

    def _ws_url(self, path):
        return self.get_url(path).replace("http://", "ws://")

    @gen_test
    def test_deny_upgrade_uses_http_error_exit(self):
        with self.assertRaises(HTTPError) as cm:
            yield websocket_connect(self._ws_url("/deny"))
        self.assertEqual(cm.exception.code, 403)
        self.assertFalse(self.rec["ws_open_called"])

    @gen_test
    def test_honor_ws_works(self):
        ws = yield websocket_connect(self._ws_url("/echo"))
        with contextlib.closing(ws):
            ws.write_message("hi")
            message = yield ws.read_message()
            self.assertEqual(message, "hi")

    @gen_test
    def test_close_on_open_exception(self):
        with ExpectLog(app_log, "Uncaught exception"):
            with ExpectLog(gen_log, ".*close_on_open_exc"):
                ws = yield websocket_connect(self._ws_url("/open_boom"))
                message = yield ws.read_message()
        self.assertIsNone(message)
        self.assertEqual(ws.close_code, 1011)
        ws.close()

    @gen_test
    def test_control_open_exception_aborts_without_close_frame(self):
        with ExpectLog(app_log, "Uncaught exception", required=False):
            ws = yield websocket_connect(self._ws_url("/open_boom_honor"))
            message = yield ws.read_message()
        self.assertIsNone(message)
        self.assertIsNone(ws.close_code)
        ws.close()

    @gen_test
    def test_close_on_first_message_exception(self):
        with ExpectLog(app_log, "Uncaught exception"):
            with ExpectLog(gen_log, ".*close_on_open_exc"):
                ws = yield websocket_connect(self._ws_url("/msg_boom"))
                ws.write_message("boom")
                message = yield ws.read_message()
        self.assertIsNone(message)
        self.assertEqual(ws.close_code, 1011)
        ws.close()

    @gen_test
    def test_control_first_message_exception(self):
        with ExpectLog(app_log, "Uncaught exception", required=False):
            ws = yield websocket_connect(self._ws_url("/msg_boom_honor"))
            ws.write_message("boom")
            message = yield ws.read_message()
        self.assertIsNone(message)
        self.assertIsNone(ws.close_code)
        ws.close()

    @gen_test
    def test_late_deny_push_is_rejected_but_session_proceeds(self):
        ws = yield websocket_connect(self._ws_url("/late_deny"))
        with contextlib.closing(ws):
            ws.write_message("still-here")
            message = yield ws.read_message()
            self.assertEqual(message, "still-here")
        self.assertTrue(self.rec["deny_phase_rejected"])


# ---------------------------------------------------------------------------
# conn_continue
# ---------------------------------------------------------------------------


class OkHandler(RequestHandler):
    def get(self):
        self.finish(b"ok")


class ConnContinueTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = _BodyRecorder()
        return Application(
            [
                ("/ok", OkHandler),
                ("/force", ForceCloseHandler),
                ("/late_force", LateForceCloseHandler),
            ],
            rec=self.rec,
        )

    @gen_test
    def test_control_two_requests_keepalive(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        try:
            for _ in range(2):
                stream.write(b"GET /ok HTTP/1.1\r\nHost: x\r\n\r\n")
                status = yield stream.read_until(b"\r\n")
                self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)
                yield stream.read_until(b"\r\n\r\n")
                body = yield stream.read_bytes(2)
                self.assertEqual(body, b"ok")
        finally:
            stream.close()

    @gen_test
    def test_force_close_stops_next_request_on_same_connection(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        try:
            stream.write(b"GET /force HTTP/1.1\r\nHost: x\r\n\r\n")
            status = yield stream.read_until(b"\r\n")
            self.assertTrue(status.startswith(b"HTTP/1.1 200"), status)
            header_bytes = yield stream.read_until(b"\r\n\r\n")
            headers = HTTPHeaders.parse(header_bytes.decode("latin1"))
            self.assertEqual(headers.get("Connection"), "close")
            body = yield stream.read_bytes(int(headers["Content-Length"]))
            self.assertEqual(body, b"closing")

            # The next request must not be processed on this connection.
            stream.write(b"GET /ok HTTP/1.1\r\nHost: x\r\n\r\n")
            rest = yield stream.read_until_close()
            self.assertEqual(rest, b"")
        finally:
            stream.close()

    def test_late_force_close_push_is_rejected(self):
        response = self.fetch("/late_force")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b"late-force-close-ok")
        self.assertTrue(self.rec.phase_rejected)


class _StickyXHeadersServer(_FrameHTTPServer):
    frame_kwargs = {"conn_continue": inbound.ConnContinue.STICKY_XHEADERS}


class StickyXHeadersTest(AsyncHTTPTestCase):
    def get_app(self):
        self.rec = type("R", (), {"xheaders_restored": []})()
        return Application([("/proto", XProtoHandler)], rec=self.rec)

    def get_http_server(self):
        return _StickyXHeadersServer(self._app, xheaders=True)

    @gen_test
    def test_xheaders_do_not_leak_to_next_request(self):
        stream = IOStream(socket.socket())
        yield stream.connect(("127.0.0.1", self.get_http_port()))
        try:
            # First request claims https via the proxy header.
            stream.write(
                b"GET /proto HTTP/1.1\r\n"
                b"Host: x\r\n"
                b"X-Forwarded-Proto: https\r\n\r\n"
            )
            yield stream.read_until(b"\r\n")
            header_bytes = yield stream.read_until(b"\r\n\r\n")
            headers = HTTPHeaders.parse(header_bytes.decode("latin1"))
            body1 = yield stream.read_bytes(int(headers["Content-Length"]))
            self.assertEqual(json_decode(body1)["protocol"], "https")

            # Second request on the same keep-alive connection sends
            # nothing; it must see the original http protocol view.
            stream.write(b"GET /proto HTTP/1.1\r\nHost: x\r\n\r\n")
            yield stream.read_until(b"\r\n")
            header_bytes = yield stream.read_until(b"\r\n\r\n")
            headers = HTTPHeaders.parse(header_bytes.decode("latin1"))
            body2 = yield stream.read_bytes(int(headers["Content-Length"]))
            self.assertEqual(json_decode(body2)["protocol"], "http")
        finally:
            stream.close()
        self.assertTrue(
            all(self.rec.xheaders_restored),
            f"xheaders restore not observed: {self.rec.xheaders_restored}",
        )


class StickyProxyAdapterUnitTest(TestCase):
    def _make_context(self):
        class Context:
            remote_ip = "127.0.0.1"
            protocol = "http"
            _orig_remote_ip = "127.0.0.1"
            _orig_protocol = "http"
            trusted_downstream = set()

            def _apply_xheaders(self, headers):
                proto = headers.get("X-Forwarded-Proto")
                if proto in ("http", "https"):
                    self.protocol = proto

            def _unapply_xheaders(self):
                self.protocol = self._orig_protocol
                self.remote_ip = self._orig_remote_ip

        return Context()

    def test_sticky_frame_restores_even_when_inner_finish_raises(self):
        context = self._make_context()

        class Connection:
            pass

        connection = Connection()
        connection.context = context
        stack = inbound.bind_policy_stack(context)
        stack.begin_segment()
        stack.push(conn_continue=inbound.ConnContinue.STICKY_XHEADERS)

        class RaisingDelegate(HTTPMessageDelegate):
            def headers_received(self, start_line, headers):
                return None

            def finish(self):
                raise RuntimeError("boom in finish")

            def on_connection_close(self):
                pass

        adapter = _ProxyAdapter(RaisingDelegate(), connection)
        adapter.headers_received(None, HTTPHeaders({"X-Forwarded-Proto": "https"}))
        self.assertEqual(context.protocol, "https")
        with self.assertRaises(RuntimeError):
            adapter.finish()
        # try/finally restoration despite the inner failure.
        self.assertEqual(context.protocol, "http")
        self.assertTrue(stack.segment.xheaders_restored)


# ---------------------------------------------------------------------------
# Concurrency isolation
# ---------------------------------------------------------------------------


class _IsoState:
    def __init__(self):
        self.arrived = {"red": Event(), "blue": Event()}
        self.release = Event()
        self.records = {}


class IsolatedStackHandler(RequestHandler):
    async def get(self):
        state = self.settings["iso"]
        color = self.get_query_argument("c")
        stack = inbound.policy_stack(self)
        fault = (
            inbound.WriteFault.FAIL_FLUSH
            if color == "red"
            else inbound.WriteFault.DROP_ON_ERROR
        )
        with stack.frame(write_fault=fault, tag=color):
            eff = self.inbound_effective
            state.records[color] = (
                id(stack),
                eff.write_fault,
            )
            state.arrived[color].set()
            await state.release.wait()
        self.finish(("ok-" + color).encode())


class ConcurrentStackIsolationTest(AsyncHTTPTestCase):
    def get_app(self):
        self.state = _IsoState()
        return Application([("/iso", IsolatedStackHandler)], iso=self.state)

    @gen_test
    def test_two_concurrent_handlers_do_not_share_stack(self):
        client = self.get_http_client()

        async def fetch(color):
            return await client.fetch(self.get_url("/iso?c=" + color))

        red_task = asyncio.ensure_future(fetch("red"))
        blue_task = asyncio.ensure_future(fetch("blue"))

        yield self.state.arrived["red"].wait()
        yield self.state.arrived["blue"].wait()
        # Both frames are live concurrently at this point.
        self.state.release.set()

        red_response = yield red_task
        blue_response = yield blue_task
        self.assertEqual(red_response.code, 200)
        self.assertEqual(blue_response.code, 200)
        self.assertEqual(red_response.body, b"ok-red")
        self.assertEqual(blue_response.body, b"ok-blue")

        red_stack_id, red_fault = self.state.records["red"]
        blue_stack_id, blue_fault = self.state.records["blue"]
        self.assertNotEqual(red_stack_id, blue_stack_id)
        self.assertIs(red_fault, inbound.WriteFault.FAIL_FLUSH)
        self.assertIs(blue_fault, inbound.WriteFault.DROP_ON_ERROR)
