# Tests for the scoped outbound request policy stack
# (tornado.httpclient_policy).
#
# These tests exercise both outbound exits through their common entry point
# (AsyncHTTPClient.fetch): SimpleAsyncHTTPClient directly, and
# CurlAsyncHTTPClient when pycurl is installed. The point under test is not
# ordinary client behavior but the difference a pushed frame makes, and that
# independent scopes on one IOLoop never share a stack.
import asyncio
import os
import socket
from contextlib import closing

from tornado import gen
from tornado.httpclient import HTTPRequest
from tornado.httpclient_policy import (
    FetchAdmit,
    OutboundFetchDenied,
    OutboundPolicyConflict,
    OutboundQueueFull,
    current_outbound_policy,
    outbound_policy,
    push_outbound_policy,
    pop_outbound_policy,
)
from tornado.simple_httpclient import HTTPTimeoutError, SimpleAsyncHTTPClient
from tornado.test.httpclient_test import CountdownHandler, HelloWorldHandler
from tornado.test.simple_httpclient_test import TriggerHandler
from tornado.test.util import AsyncHTTPTestCase, refusing_port
from tornado.testing import gen_test
from tornado.web import Application, url

# Timers are slow on the Windows CI machines; mirror simple_httpclient_test.
SHORT = 0.5 if os.name == "nt" else 0.15
LONG = 3600.0


class OutboundPolicyTestCase(AsyncHTTPTestCase):
    def get_app(self):
        # callable objects to finish pending /trigger requests
        self.triggers = []
        return Application(
            [
                url(
                    r"/trigger",
                    TriggerHandler,
                    dict(queue=self.triggers, wake_callback=self.stop),
                ),
                url(r"/hello", HelloWorldHandler),
                url(r"/countdown/([0-9]+)", CountdownHandler, name="countdown"),
            ]
        )

    def get_http_client(self):
        # The policy stack is client-independent; tests build their own
        # force_instance clients, but the case still needs one for setup.
        return SimpleAsyncHTTPClient(force_instance=True)

    def new_client(self, **kwargs):
        kwargs.setdefault("force_instance", True)
        return SimpleAsyncHTTPClient(**kwargs)

    async def wait_for_trigger(self):
        while not self.triggers:
            await gen.sleep(0.01)

    def release_trigger(self):
        self.triggers.pop(0)()

    async def flush_trigger_cleanup(self):
        # Let server-side coroutines of client-timed-out requests finish
        # closing their streams so their logs don't land in another test.
        await gen.sleep(0.3)

    # ---- empty stack / pin behavior -------------------------------------

    @gen_test
    async def test_empty_stack_is_pin_behavior(self):
        with closing(self.new_client()) as client:
            response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)
            self.assertEqual(response.body, b"Hello world!")
            self.assertIsNone(current_outbound_policy())

    @gen_test
    async def test_honor_frame_is_pin_behavior(self):
        with closing(self.new_client()) as client:
            with outbound_policy():
                response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)
            self.assertIsNone(current_outbound_policy())

    # ---- fetch_admit ------------------------------------------------------

    @gen_test
    async def test_deny_new_rejects_and_restores(self):
        with closing(self.new_client()) as client:
            with outbound_policy(fetch_admit="deny_new"):
                with self.assertRaises(OutboundFetchDenied) as cm:
                    await client.fetch(self.get_url("/hello"))
                self.assertEqual(cm.exception.gate, "deny_new")
                self.assertEqual(599, cm.exception.code)
                # Nothing was admitted.
                self.assertEqual(0, len(client.queue))
                self.assertEqual(0, len(client.active))
            # Popping restores ordinary behavior.
            response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)

    @gen_test
    async def test_deny_new_string_value_accepted(self):
        with closing(self.new_client()) as client:
            with outbound_policy(fetch_admit=FetchAdmit.DENY_NEW):
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))

    @gen_test
    async def test_deny_new_leaves_in_flight_request_alone(self):
        with closing(self.new_client()) as client:
            fut = client.fetch(
                self.get_url("/trigger?wake=false"), request_timeout=LONG
            )
            await self.wait_for_trigger()
            # Push deny_new after transmission started; the admitted fetch
            # must neither be cancelled nor rewritten.
            with outbound_policy(fetch_admit="deny_new"):
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))
                self.release_trigger()
                response = await fut
            # /trigger has an empty body by design; the point is a normal
            # 200 completion rather than a fabricated response.
            self.assertEqual(200, response.code)

    @gen_test
    async def test_one_more_admits_exactly_one(self):
        with closing(self.new_client()) as client:
            with outbound_policy(fetch_admit="one_more"):
                response = await client.fetch(self.get_url("/hello"))
                self.assertEqual(200, response.code)
                with self.assertRaises(OutboundFetchDenied) as cm:
                    await client.fetch(self.get_url("/hello"))
                self.assertEqual(cm.exception.gate, "one_more")
                # Third fetch stays out too.
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))
            # The permit is per-frame; a fresh frame gets its own.
            with outbound_policy(fetch_admit="one_more"):
                response = await client.fetch(self.get_url("/hello"))
                self.assertEqual(200, response.code)

    @gen_test
    async def test_one_more_rereads_live_stack_in_nested_scope(self):
        with closing(self.new_client()) as client:
            with outbound_policy(fetch_admit="one_more"):
                await client.fetch(self.get_url("/hello"))
                # A nested one_more frame grants its own permit: the second
                # fetch re-reads the then-current top instead of reusing the
                # first fetch's decision.
                with outbound_policy(fetch_admit="one_more"):
                    response = await client.fetch(self.get_url("/hello"))
                    self.assertEqual(200, response.code)
                    with self.assertRaises(OutboundFetchDenied):
                        await client.fetch(self.get_url("/hello"))
                # Inner frame popped: the exhausted outer governs again.
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))

    @gen_test
    async def test_concurrent_scopes_do_not_cross_admit(self):
        with closing(self.new_client()) as client:
            async def denied():
                with outbound_policy(fetch_admit="deny_new"):
                    return await client.fetch(self.get_url("/hello"))

            async def allowed():
                # This task carries no frame; it must keep using the plain
                # configured API even while the other task is denied.
                return await client.fetch(self.get_url("/hello"))

            denied_result, allowed_response = await asyncio.gather(
                denied(), allowed(), return_exceptions=True
            )
            self.assertIsInstance(denied_result, OutboundFetchDenied)
            self.assertEqual(200, allowed_response.code)
            self.assertIsNone(current_outbound_policy())

    # ---- queue_gate --------------------------------------------------------

    @gen_test
    async def test_fail_fast_when_full_does_not_queue(self):
        with closing(self.new_client(max_clients=1)) as client:
            fut1 = client.fetch(
                self.get_url("/trigger?wake=false"), request_timeout=LONG
            )
            await self.wait_for_trigger()
            self.assertEqual(1, len(client.active))
            with outbound_policy(queue_gate="fail_fast"):
                with self.assertRaises(OutboundQueueFull) as cm:
                    await client.fetch(
                        self.get_url("/hello"), connect_timeout=SHORT
                    )
                self.assertEqual(cm.exception.gate, "fail_fast")
                self.assertEqual(599, cm.exception.code)
                # It never took a queue slot.
                self.assertEqual(0, len(client.queue))
                self.assertEqual(1, len(client.active))
            self.release_trigger()
            response = await fut1
            self.assertEqual(200, response.code)

    @gen_test
    async def test_fail_fast_with_free_capacity_starts_normally(self):
        with closing(self.new_client(max_clients=2)) as client:
            with outbound_policy(queue_gate="fail_fast"):
                response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)
            self.assertEqual(0, len(client.queue))

    @gen_test
    async def test_passthrough_keeps_queueing(self):
        with closing(self.new_client(max_clients=1)) as client:
            fut1 = client.fetch(
                self.get_url("/trigger?wake=false"), request_timeout=LONG
            )
            await self.wait_for_trigger()
            with outbound_policy(queue_gate="passthrough"):
                queued = client.fetch(self.get_url("/hello"))
                await gen.sleep(0.05)
                self.assertEqual(1, len(client.queue))
            self.release_trigger()
            self.assertEqual(200, (await fut1).code)
            self.assertEqual(200, (await queued).code)

    @gen_test
    async def test_fail_fast_isolated_from_other_scope(self):
        with closing(self.new_client(max_clients=1)) as client:
            fut1 = client.fetch(
                self.get_url("/trigger?wake=false"), request_timeout=LONG
            )
            await self.wait_for_trigger()

            async def fast_fail():
                with outbound_policy(queue_gate="fail_fast"):
                    return await client.fetch(self.get_url("/hello"))

            async def ordinary_queue():
                # No frame here: queueing must keep working.
                return await client.fetch(self.get_url("/hello"))

            failed_task = asyncio.ensure_future(fast_fail())
            queued_task = asyncio.ensure_future(ordinary_queue())
            await asyncio.wait({failed_task})
            failed = failed_task.exception()
            self.assertIsInstance(failed, OutboundQueueFull)
            self.assertEqual(failed.gate, "fail_fast")
            # Let the queued (unframed) request proceed.
            self.release_trigger()
            queued_response = await queued_task
            self.assertEqual(200, queued_response.code)
            self.assertEqual(200, (await fut1).code)

    # ---- timeout_gate ------------------------------------------------------

    @gen_test
    async def test_shrink_now_tightens_request_timeout(self):
        with closing(self.new_client()) as client:
            with outbound_policy(
                timeout_gate="shrink_now", request_timeout=SHORT
            ):
                with self.assertRaises(HTTPTimeoutError):
                    await client.fetch(
                        self.get_url("/trigger?wake=false"),
                        request_timeout=LONG,
                    )
            self.release_trigger()
            await self.flush_trigger_cleanup()

    @gen_test
    async def test_shrink_now_takes_min_with_request_field(self):
        with closing(self.new_client()) as client:
            with outbound_policy(
                timeout_gate="shrink_now", request_timeout=SHORT
            ):
                # A looser per-request timeout must not relax the frame.
                with self.assertRaises(HTTPTimeoutError):
                    await client.fetch(
                        self.get_url("/trigger?wake=false"),
                        request_timeout=LONG,
                    )
            self.release_trigger()
            await self.flush_trigger_cleanup()

    @gen_test
    async def test_shrink_now_timeout_uses_ordinary_failure_surface(self):
        # Even with raise_error=False a (non-HTTP) timeout still surfaces
        # through the ordinary client failure path.
        with closing(self.new_client()) as client:
            with outbound_policy(
                timeout_gate="shrink_now", request_timeout=SHORT
            ):
                with self.assertRaises(HTTPTimeoutError):
                    await client.fetch(
                        self.get_url("/trigger?wake=false"),
                        request_timeout=LONG,
                        raise_error=False,
                    )
            self.release_trigger()
            await self.flush_trigger_cleanup()

    @gen_test
    async def test_freeze_prevents_request_timeout_failure(self):
        with closing(self.new_client()) as client:
            with outbound_policy(timeout_gate="freeze"):
                fut = client.fetch(
                    self.get_url("/trigger?wake=false"),
                    request_timeout=SHORT,
                )
                await self.wait_for_trigger()
                # Wait well past the configured request timeout.
                await gen.sleep(SHORT * 3)
                self.release_trigger()
                response = await fut
            # /trigger has an empty body by design; the point is a normal
            # 200 completion rather than a fabricated response.
            self.assertEqual(200, response.code)

    @gen_test
    async def test_freeze_does_not_swallow_real_network_failure(self):
        cleanup_func, port = refusing_port()
        self.addCleanup(cleanup_func)
        with closing(self.new_client()) as client:
            with outbound_policy(timeout_gate="freeze"):
                with self.assertRaises(socket.error):
                    await client.fetch(
                        "http://127.0.0.1:%d/" % port,
                        request_timeout=LONG,
                        connect_timeout=LONG,
                    )

    @gen_test
    async def test_freeze_late_push_with_ticking_timer_is_rejected(self):
        with closing(self.new_client()) as client:
            # An outer (honor) frame owns the in-flight request and its
            # running timer.
            with outbound_policy():
                fut = client.fetch(
                    self.get_url("/trigger?wake=false"), request_timeout=LONG
                )
                await self.wait_for_trigger()
                # The whole-request timeout is now armed.
                with self.assertRaises(OutboundPolicyConflict):
                    with outbound_policy(timeout_gate="freeze"):
                        pass
                # Nothing was pushed; the outer frame still governs.
                self.assertEqual(1, len(current_outbound_policy()))
                self.release_trigger()
                self.assertEqual(200, (await fut).code)

    @gen_test
    async def test_freeze_does_not_reach_request_admitted_before_frame(self):
        # A request admitted with an empty stack keeps its own timeout;
        # pushing freeze later is allowed (no active frame owned the timer)
        # but does not retroactively freeze the pre-frame request.
        with closing(self.new_client()) as client:
            fut = client.fetch(
                self.get_url("/trigger?wake=false"), request_timeout=SHORT
            )
            await self.wait_for_trigger()
            with outbound_policy(timeout_gate="freeze"):
                await gen.sleep(SHORT * 3)
                with self.assertRaises(HTTPTimeoutError):
                    await fut
            self.release_trigger()
            await self.flush_trigger_cleanup()

    @gen_test
    async def test_concurrent_scopes_do_not_cross_timeout_gate(self):
        with closing(self.new_client(max_clients=4)) as client:
            async def frozen():
                with outbound_policy(timeout_gate="freeze"):
                    return await client.fetch(
                        self.get_url("/trigger?wake=false"),
                        request_timeout=SHORT,
                    )

            async def shrunk():
                await gen.sleep(0.05)
                with outbound_policy(
                    timeout_gate="shrink_now", request_timeout=SHORT
                ):
                    try:
                        return await client.fetch(
                            self.get_url("/trigger?wake=false"),
                            request_timeout=LONG,
                        )
                    except HTTPTimeoutError as err:
                        return err

            frozen_task = asyncio.ensure_future(frozen())
            shrunk_task = asyncio.ensure_future(shrunk())
            # Wait until both trigger handlers are hanging.
            while len(self.triggers) < 2:
                await gen.sleep(0.01)
            # The shrunk scope fails on its own tightened timer; this must
            # not touch the frozen scope's request.
            shrunk_result = await shrunk_task
            self.assertIsInstance(shrunk_result, HTTPTimeoutError)
            self.assertFalse(frozen_task.done())
            # Frozen request was registered first; release it.
            self.triggers.pop(0)()
            frozen_response = await frozen_task
            self.assertEqual(200, frozen_response.code)
            # Clean up the shrunk request's hanging server coroutine.
            self.triggers.pop(0)()
            await self.flush_trigger_cleanup()

    # ---- nesting, cleanup and invariants ----------------------------------

    @gen_test
    async def test_nested_frames_restore_outer_on_pop(self):
        with closing(self.new_client()) as client:
            with outbound_policy(fetch_admit="one_more"):
                await client.fetch(self.get_url("/hello"))
                with outbound_policy(fetch_admit="deny_new"):
                    with self.assertRaises(OutboundFetchDenied) as cm:
                        await client.fetch(self.get_url("/hello"))
                    self.assertEqual(cm.exception.gate, "deny_new")
                # Inner popped: exhausted one_more frame is back on top.
                with self.assertRaises(OutboundFetchDenied) as cm:
                    await client.fetch(self.get_url("/hello"))
                self.assertEqual(cm.exception.gate, "one_more")
            # Fully unwound: configured behavior again.
            response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)

    @gen_test
    async def test_exception_pops_the_frame(self):
        with closing(self.new_client()) as client:
            with self.assertRaises(RuntimeError):
                with outbound_policy(fetch_admit="deny_new"):
                    raise RuntimeError("boom")
            self.assertIsNone(current_outbound_policy())
            response = await client.fetch(self.get_url("/hello"))
            self.assertEqual(200, response.code)

    @gen_test
    async def test_push_pop_token_api(self):
        token = push_outbound_policy(fetch_admit="deny_new")
        try:
            self.assertEqual(1, len(current_outbound_policy()))
            with closing(self.new_client()) as client:
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))
        finally:
            pop_outbound_policy(token)
        self.assertIsNone(current_outbound_policy())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            push_outbound_policy(fetch_admit="nonsense")
        with self.assertRaises(ValueError):
            push_outbound_policy(
                timeout_gate="shrink_now"
            )
        with self.assertRaises(ValueError):
            push_outbound_policy(
                timeout_gate="freeze", request_timeout=1.0
            )
        with self.assertRaises(ValueError):
            push_outbound_policy(
                timeout_gate="shrink_now", request_timeout=0
            )
        self.assertIsNone(current_outbound_policy())

    @gen_test
    async def test_permanent_knobs_are_not_mutated(self):
        with closing(self.new_client(max_clients=3)) as client:
            defaults_snapshot = dict(client.defaults)
            max_clients = client.max_clients
            with outbound_policy(
                fetch_admit="deny_new",
                timeout_gate="shrink_now",
                request_timeout=SHORT,
                queue_gate="fail_fast",
            ):
                with self.assertRaises(OutboundFetchDenied):
                    await client.fetch(self.get_url("/hello"))
            self.assertEqual(max_clients, client.max_clients)
            self.assertEqual(defaults_snapshot, client.defaults)
        # HTTPRequest defaults are untouched too.
        self.assertEqual(20.0, HTTPRequest._DEFAULTS["request_timeout"])
        self.assertEqual(20.0, HTTPRequest._DEFAULTS["connect_timeout"])
        self.assertTrue(HTTPRequest._DEFAULTS["follow_redirects"])

    # ---- unrelated axes stay pinned ---------------------------------------

    @gen_test
    async def test_follow_redirects_field_stays_pinned(self):
        with closing(self.new_client()) as client:
            with outbound_policy(queue_gate="fail_fast"):
                response = await client.fetch(
                    self.get_url("/countdown/3"),
                    follow_redirects=False,
                    raise_error=False,
                )
            self.assertEqual(302, response.code)

    @gen_test
    async def test_redirect_chain_inside_deny_frame_completes(self):
        # The fetch was admitted before the frame; its internal redirect
        # continuations must not be re-gated by deny_new.
        with closing(self.new_client()) as client:
            fut = client.fetch(self.get_url("/countdown/5"))
            await gen.sleep(SHORT)
            with outbound_policy(fetch_admit="deny_new"):
                response = await fut
            self.assertEqual(200, response.code)
            self.assertTrue(response.effective_url.endswith("/countdown/0"))


# A small parallel run against the curl-backed exit. Everything interesting
# lives above the implementation, but the curl queue/timeout paths differ, so
# repeat the frame-observable checks when pycurl is available.
try:
    import pycurl  # noqa: F401

    from tornado.curl_httpclient import CurlAsyncHTTPClient

    class CurlOutboundPolicyTestCase(AsyncHTTPTestCase):
        def get_app(self):
            self.triggers = []
            return Application(
                [
                    url(
                        r"/trigger",
                        TriggerHandler,
                        dict(queue=self.triggers, wake_callback=self.stop),
                    ),
                    url(r"/hello", HelloWorldHandler),
                    url(r"/countdown/([0-9]+)", CountdownHandler, name="countdown"),
                ]
            )

        def get_http_client(self):
            return CurlAsyncHTTPClient(force_instance=True)

        def new_client(self, **kwargs):
            kwargs.setdefault("force_instance", True)
            return CurlAsyncHTTPClient(**kwargs)

        async def wait_for_trigger(self):
            while not self.triggers:
                await gen.sleep(0.01)

        @gen_test
        async def test_curl_deny_new(self):
            with closing(self.new_client()) as client:
                with outbound_policy(fetch_admit="deny_new"):
                    with self.assertRaises(OutboundFetchDenied):
                        await client.fetch(self.get_url("/hello"))
                self.assertEqual(200, (await client.fetch(self.get_url("/hello"))).code)

        @gen_test
        async def test_curl_fail_fast(self):
            with closing(self.new_client(max_clients=1)) as client:
                fut1 = client.fetch(
                    self.get_url("/trigger?wake=false"), request_timeout=LONG
                )
                await self.wait_for_trigger()
                with outbound_policy(queue_gate="fail_fast"):
                    with self.assertRaises(OutboundQueueFull):
                        await client.fetch(self.get_url("/hello"))
                self.triggers.pop(0)()
                self.assertEqual(200, (await fut1).code)

        @gen_test
        async def test_curl_fail_fast_free_capacity(self):
            with closing(self.new_client()) as client:
                with outbound_policy(queue_gate="fail_fast"):
                    response = await client.fetch(self.get_url("/hello"))
                self.assertEqual(200, response.code)

        @gen_test
        async def test_curl_freeze_timeout(self):
            with closing(self.new_client()) as client:
                with outbound_policy(timeout_gate="freeze"):
                    fut = client.fetch(
                        self.get_url("/trigger?wake=false"),
                        request_timeout=SHORT,
                    )
                    await self.wait_for_trigger()
                    await gen.sleep(SHORT * 3)
                    self.triggers.pop(0)()
                    response = await fut
                self.assertEqual(200, response.code)

        @gen_test
        async def test_curl_shrink_timeout_surface(self):
            with closing(self.new_client()) as client:
                with outbound_policy(
                    timeout_gate="shrink_now", request_timeout=SHORT
                ):
                    with self.assertRaises(Exception):
                        await client.fetch(
                            self.get_url("/trigger?wake=false"),
                            request_timeout=LONG,
                        )
                self.triggers.pop(0)()

        @gen_test
        async def test_curl_redirect_pin(self):
            with closing(self.new_client()) as client:
                with outbound_policy(queue_gate="fail_fast"):
                    response = await client.fetch(
                        self.get_url("/countdown/3"),
                        follow_redirects=False,
                        raise_error=False,
                    )
                self.assertEqual(302, response.code)

except ImportError:
    pass
