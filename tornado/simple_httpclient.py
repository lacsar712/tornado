import base64
import collections
import copy
import functools
import re
import socket
import ssl
import sys
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from io import BytesIO
from types import TracebackType
from typing import Any, Optional, Type, cast

from tornado import gen, httputil, version
from tornado.escape import _unicode
from tornado.http1connection import HTTP1Connection, HTTP1ConnectionParameters
from tornado.httpclient import (
    AsyncHTTPClient,
    ConnectionFollowScope,
    HTTPClientConnectionFollowError,
    HTTPError,
    HTTPRequest,
    HTTPResponse,
    _FollowedConnection,
    _RequestProxy,
    _current_follow_scope,
    _scope_fail_response,
    _scope_validate_request,
    main,
)
from tornado.ioloop import IOLoop
from tornado.iostream import IOStream, StreamClosedError
from tornado.log import gen_log
from tornado.netutil import (
    OverrideResolver,
    Resolver,
    _client_ssl_defaults,
    is_valid_ip,
)
from tornado.tcpclient import TCPClient


class HTTPTimeoutError(HTTPError):
    """Error raised by SimpleAsyncHTTPClient on timeout.

    For historical reasons, this is a subclass of `.HTTPClientError`
    which simulates a response code of 599.

    .. versionadded:: 5.1
    """

    def __init__(self, message: str) -> None:
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or "Timeout"


class HTTPStreamClosedError(HTTPError):
    """Error raised by SimpleAsyncHTTPClient when the underlying stream is closed.

    When a more specific exception is available (such as `ConnectionResetError`),
    it may be raised instead of this one.

    For historical reasons, this is a subclass of `.HTTPClientError`
    which simulates a response code of 599.

    .. versionadded:: 5.1
    """

    def __init__(self, message: str) -> None:
        super().__init__(599, message=message)

    def __str__(self) -> str:
        return self.message or "Stream closed"


class SimpleAsyncHTTPClient(AsyncHTTPClient):
    """Non-blocking HTTP client with no external dependencies.

    This class implements an HTTP 1.1 client on top of Tornado's IOStreams.
    Some features found in the curl-based AsyncHTTPClient are not yet
    supported.  In particular, proxies are not supported, connections
    are not reused, and callers cannot select the network interface to be
    used.

    This implementation supports the following arguments, which can be passed
    to ``configure()`` to control the global singleton, or to the constructor
    when ``force_instance=True``.

    ``max_clients`` is the number of concurrent requests that can be
    in progress; when this limit is reached additional requests will be
    queued. Note that time spent waiting in this queue still counts
    against the ``request_timeout``.

    ``defaults`` is a dict of parameters that will be used as defaults on all
    `.HTTPRequest` objects submitted to this client.

    ``hostname_mapping`` is a dictionary mapping hostnames to IP addresses.
    It can be used to make local DNS changes when modifying system-wide
    settings like ``/etc/hosts`` is not possible or desirable (e.g. in
    unittests). ``resolver`` is similar, but using the `.Resolver` interface
    instead of a simple mapping.

    ``max_buffer_size`` (default 100MB) is the number of bytes
    that can be read into memory at once. ``max_body_size``
    (defaults to ``max_buffer_size``) is the largest response body
    that the client will accept.  Without a
    ``streaming_callback``, the smaller of these two limits
    applies; with a ``streaming_callback`` only ``max_body_size``
    does.

    .. versionchanged:: 4.2
        Added the ``max_body_size`` argument.
    """

    def initialize(  # type: ignore
        self,
        max_clients: int = 10,
        hostname_mapping: dict[str, str] | None = None,
        max_buffer_size: int = 104857600,
        resolver: Resolver | None = None,
        defaults: dict[str, Any] | None = None,
        max_header_size: int | None = None,
        max_body_size: int | None = None,
    ) -> None:
        super().initialize(defaults=defaults)
        self.max_clients = max_clients
        self.queue: collections.deque[
            tuple[
                object,
                HTTPRequest,
                Callable[[HTTPResponse], None],
                ConnectionFollowScope | None,
                tuple[Any, ...] | None,
            ]
        ] = collections.deque()
        self.active: dict[
            object, tuple[HTTPRequest, Callable[[HTTPResponse], None]]
        ] = {}
        self.waiting: dict[
            object,
            tuple[
                HTTPRequest,
                Callable[[HTTPResponse], None],
                object,
                ConnectionFollowScope | None,
                tuple[Any, ...] | None,
            ],
        ] = {}
        self.max_buffer_size = max_buffer_size
        self.max_header_size = max_header_size
        self.max_body_size = max_body_size
        # TCPClient could create a Resolver for us, but we have to do it
        # ourselves to support hostname_mapping.
        if resolver:
            self.resolver = resolver
            self.own_resolver = False
        else:
            self.resolver = Resolver()
            self.own_resolver = True
        if hostname_mapping is not None:
            self.resolver = OverrideResolver(
                resolver=self.resolver, mapping=hostname_mapping
            )
        self.tcp_client = TCPClient(resolver=self.resolver)

    def close(self) -> None:
        super().close()
        if self.own_resolver:
            self.resolver.close()
        self.tcp_client.close()

    @staticmethod
    def _follow_fingerprint(request: HTTPRequest) -> tuple[Any, ...]:
        """Identifies everything that must stay constant for two requests to
        share a pinned connection: target endpoint, proxy, and TLS setup."""
        parsed = urllib.parse.urlsplit(_unicode(request.url))
        netloc = parsed.netloc
        if "@" in netloc:
            _userpass, _, netloc = netloc.rpartition("@")
        host, port = httputil.split_host_and_port(netloc)
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        if re.match(r"^\[.*\]$", host):
            # raw ipv6 addresses in urls are enclosed in brackets
            host = host[1:-1]
        ssl_options = request.ssl_options
        return (
            parsed.scheme,
            host,
            port,
            request.proxy_host,
            request.proxy_port,
            request.proxy_username,
            request.proxy_auth_mode,
            request.validate_cert,
            request.ca_certs,
            request.client_cert,
            request.client_key,
            # ssl_options may be an unhashable dict; identity is enough.
            id(ssl_options) if ssl_options is not None else None,
            request.network_interface,
            request.allow_ipv6,
        )

    def fetch_impl(
        self, request: HTTPRequest, callback: Callable[[HTTPResponse], None]
    ) -> None:
        key = object()
        scope: ConnectionFollowScope | None = None
        fingerprint: tuple[Any, ...] | None = None
        # Snapshot the follow scope from the caller's context now; the
        # request may sit in the queue before it actually runs.
        scope = _current_follow_scope()
        if scope is not None:
            fingerprint = self._follow_fingerprint(request)
            try:
                _scope_validate_request(self, request, fingerprint, scope=scope)
            except HTTPClientConnectionFollowError as e:
                _scope_fail_response(self, request, e, callback)
                return
        self.queue.append((key, request, callback, scope, fingerprint))
        assert request.connect_timeout is not None
        assert request.request_timeout is not None
        timeout_handle = None
        if len(self.active) >= self.max_clients:
            timeout = (
                min(request.connect_timeout, request.request_timeout)
                or request.connect_timeout
                or request.request_timeout
            )  # min but skip zero
            if timeout:
                timeout_handle = self.io_loop.add_timeout(
                    self.io_loop.time() + timeout,
                    functools.partial(self._on_timeout, key, "in request queue"),
                )
        self.waiting[key] = (request, callback, timeout_handle, scope, fingerprint)
        self._process_queue()
        if self.queue:
            gen_log.debug(
                "max_clients limit reached, request queued. "
                "%d active, %d queued requests." % (len(self.active), len(self.queue))
            )

    def _process_queue(self) -> None:
        while self.queue and len(self.active) < self.max_clients:
            key, request, callback, scope, fingerprint = self.queue.popleft()
            if key not in self.waiting:
                continue
            self._remove_timeout(key)
            self.active[key] = (request, callback)
            release_callback = functools.partial(self._release_fetch, key)
            self._handle_request(
                request, release_callback, callback, scope, fingerprint
            )

    def _connection_class(self) -> type:
        return _HTTPConnection

    def _handle_request(
        self,
        request: HTTPRequest,
        release_callback: Callable[[], None],
        final_callback: Callable[[HTTPResponse], None],
        follow_scope: ConnectionFollowScope | None = None,
        follow_fingerprint: tuple[Any, ...] | None = None,
    ) -> None:
        self._connection_class()(
            self,
            request,
            release_callback,
            final_callback,
            self.max_buffer_size,
            self.tcp_client,
            self.max_header_size,
            self.max_body_size,
            follow_scope,
            follow_fingerprint,
        )

    def _release_fetch(self, key: object) -> None:
        del self.active[key]
        self._process_queue()

    def _remove_timeout(self, key: object) -> None:
        if key in self.waiting:
            request, callback, timeout_handle, _scope, _fp = self.waiting[key]
            if timeout_handle is not None:
                self.io_loop.remove_timeout(timeout_handle)
            del self.waiting[key]

    def _on_timeout(self, key: object, info: str | None = None) -> None:
        """Timeout callback of request.

        Construct a timeout HTTPResponse when a timeout occurs.

        :arg object key: A simple object to mark the request.
        :info string key: More detailed timeout information.
        """
        request, callback, timeout_handle, scope, fingerprint = self.waiting[key]
        self.queue.remove((key, request, callback, scope, fingerprint))

        error_message = f"Timeout {info}" if info else "Timeout"
        timeout_response = HTTPResponse(
            request,
            599,
            error=HTTPTimeoutError(error_message),
            request_time=self.io_loop.time() - request.start_time,
        )
        self.io_loop.add_callback(callback, timeout_response)
        del self.waiting[key]


class _HTTPConnection(httputil.HTTPMessageDelegate):
    _SUPPORTED_METHODS = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"}

    def __init__(
        self,
        client: SimpleAsyncHTTPClient | None,
        request: HTTPRequest,
        release_callback: Callable[[], None],
        final_callback: Callable[[HTTPResponse], None],
        max_buffer_size: int,
        tcp_client: TCPClient,
        max_header_size: int,
        max_body_size: int,
        follow_scope: ConnectionFollowScope | None = None,
        follow_fingerprint: tuple[Any, ...] | None = None,
    ) -> None:
        self.io_loop = IOLoop.current()
        self.start_time = self.io_loop.time()
        self.start_wall_time = time.time()
        self.client = client
        self.request = request
        self.release_callback = release_callback
        self.final_callback = final_callback
        self.max_buffer_size = max_buffer_size
        self.tcp_client = tcp_client
        self.max_header_size = max_header_size
        self.max_body_size = max_body_size
        self.code: int | None = None
        self.headers: httputil.HTTPHeaders | None = None
        self.chunks = bytearray()
        self._decompressor = None
        # Timeout handle returned by IOLoop.add_timeout
        self._timeout: object = None
        self._sockaddr = None
        self.connection: HTTP1Connection | None = None
        # ConnectionFollowScope support. When follow_scope is set the
        # connection is kept alive and bound to the scope instead of being
        # closed after the (first) request.
        self.follow_scope = follow_scope
        self.follow_fingerprint = follow_fingerprint
        self.follow_record: _FollowedConnection | None = None
        self._follow_streaming = False
        self._follow_redirect_pending = False
        self._follow_cleaned = False
        self._follow_pending_error: BaseException | None = None
        self._follow_final_reusable = False
        self._response_version: str | None = None
        IOLoop.current().add_future(
            gen.convert_yielded(self.run()), lambda f: f.result()
        )

    async def run(self) -> None:
        if self.follow_scope is not None:
            try:
                await self._run_followed()
            except Exception:
                if not self._handle_exception(*sys.exc_info()):
                    raise
            return
        try:
            self.parsed = urllib.parse.urlsplit(_unicode(self.request.url))
            if self.parsed.scheme not in ("http", "https"):
                raise ValueError("Unsupported url scheme: %s" % self.request.url)
            # urlsplit results have hostname and port results, but they
            # didn't support ipv6 literals until python 2.7.
            netloc = self.parsed.netloc
            if "@" in netloc:
                userpass, _, netloc = netloc.rpartition("@")
            host, port = httputil.split_host_and_port(netloc)
            if port is None:
                port = 443 if self.parsed.scheme == "https" else 80
            if re.match(r"^\[.*\]$", host):
                # raw ipv6 addresses in urls are enclosed in brackets
                host = host[1:-1]
            self.parsed_hostname = host  # save final host for _on_connect

            if self.request.allow_ipv6 is False:
                af = socket.AF_INET
            else:
                af = socket.AF_UNSPEC

            ssl_options = self._get_ssl_options(self.parsed.scheme)

            source_ip = None
            if self.request.network_interface:
                if is_valid_ip(self.request.network_interface):
                    source_ip = self.request.network_interface
                else:
                    raise ValueError(
                        "Unrecognized IPv4 or IPv6 address for network_interface, got %r"
                        % (self.request.network_interface,)
                    )

            if self.request.connect_timeout and self.request.request_timeout:
                timeout = min(
                    self.request.connect_timeout, self.request.request_timeout
                )
            elif self.request.connect_timeout:
                timeout = self.request.connect_timeout
            elif self.request.request_timeout:
                timeout = self.request.request_timeout
            else:
                timeout = 0
            try:
                stream = await self.tcp_client.connect(
                    host,
                    port,
                    af=af,
                    ssl_options=ssl_options,
                    max_buffer_size=self.max_buffer_size,
                    source_ip=source_ip,
                    timeout=(
                        self.start_time + timeout - self.io_loop.time()
                        if timeout
                        else None
                    ),
                )
            except TimeoutError as e:
                # TCPClient's timeout messages describe which phase of
                # the connection timed out (DNS, TCP, or TLS).
                raise HTTPTimeoutError(str(e) or "Timeout while connecting") from None

            if self.final_callback is None:
                # final_callback is cleared if we've hit our timeout.
                stream.close()
                return
            self.stream = stream
            self.stream.set_close_callback(self.on_connection_close)
            self._remove_timeout()
            if self.final_callback is None:
                return
            if self.request.request_timeout:
                self._timeout = self.io_loop.add_timeout(
                    self.start_time + self.request.request_timeout,
                    functools.partial(self._on_timeout, "during request"),
                )
            if (
                self.request.method not in self._SUPPORTED_METHODS
                and not self.request.allow_nonstandard_methods
            ):
                raise KeyError("unknown method %s" % self.request.method)
            for key in (
                "proxy_host",
                "proxy_port",
                "proxy_username",
                "proxy_password",
                "proxy_auth_mode",
            ):
                if getattr(self.request, key, None):
                    raise NotImplementedError("%s not supported" % key)
            if "Connection" not in self.request.headers:
                self.request.headers["Connection"] = "close"
            req_path = self._prepare_request_message()
            self.connection = self._create_connection(stream)
            start_line = httputil.RequestStartLine(self.request.method, req_path, "")
            self.connection.write_headers(start_line, self.request.headers)
            if self.request.expect_100_continue:
                await self.connection.read_response(self)
            else:
                await self._write_body(True)
        except Exception:
            if not self._handle_exception(*sys.exc_info()):
                raise

    def _prepare_request_message(self) -> str:
        """Fills in the request headers and returns the request path.

        Shared by the normal and follow-scope code paths. Assumes
        ``self.parsed`` has been set for the current hop.
        """
        if "Host" not in self.request.headers:
            if "@" in self.parsed.netloc:
                self.request.headers["Host"] = self.parsed.netloc.rpartition("@")[-1]
            else:
                self.request.headers["Host"] = self.parsed.netloc
        username, password = None, None
        if self.parsed.username is not None:
            username, password = self.parsed.username, self.parsed.password
        elif self.request.auth_username is not None:
            username = self.request.auth_username
            password = self.request.auth_password or ""
        if username is not None:
            assert password is not None
            if self.request.auth_mode not in (None, "basic"):
                raise ValueError("unsupported auth_mode %s" % self.request.auth_mode)
            self.request.headers["Authorization"] = "Basic " + _unicode(
                base64.b64encode(httputil.encode_username_password(username, password))
            )
        if self.request.user_agent:
            self.request.headers["User-Agent"] = self.request.user_agent
        elif self.request.headers.get("User-Agent") is None:
            self.request.headers["User-Agent"] = f"Tornado/{version}"
        if not self.request.allow_nonstandard_methods:
            # Some HTTP methods nearly always have bodies while others
            # almost never do. Fail in this case unless the user has
            # opted out of sanity checks with allow_nonstandard_methods.
            body_expected = self.request.method in ("POST", "PATCH", "PUT")
            body_present = (
                self.request.body is not None or self.request.body_producer is not None
            )
            if (body_expected and not body_present) or (
                body_present and not body_expected
            ):
                raise ValueError(
                    "Body must %sbe None for method %s (unless "
                    "allow_nonstandard_methods is true)"
                    % ("not " if body_expected else "", self.request.method)
                )
        if self.request.expect_100_continue:
            self.request.headers["Expect"] = "100-continue"
        if self.request.body is not None:
            # When body_producer is used the caller is responsible for
            # setting Content-Length (or else chunked encoding will be used).
            self.request.headers["Content-Length"] = str(len(self.request.body))
        if self.request.method == "POST" and "Content-Type" not in self.request.headers:
            self.request.headers["Content-Type"] = "application/x-www-form-urlencoded"
        if self.request.decompress_response:
            self.request.headers["Accept-Encoding"] = "gzip"
        req_path = (self.parsed.path or "/") + (
            ("?" + self.parsed.query) if self.parsed.query else ""
        )
        return req_path

    # ------------------------------------------------------------------
    # ConnectionFollowScope code path
    # ------------------------------------------------------------------

    @staticmethod
    def _follow_release_record(record: _FollowedConnection) -> None:
        # Normal rule for a simple_httpclient connection is to close it as
        # soon as the scope no longer needs it.
        stream = record.payload.get("stream")
        if stream is not None:
            try:
                stream.set_close_callback(None)
            except Exception:
                pass
            stream.close()

    async def _run_followed(self) -> None:
        assert self.follow_scope is not None
        assert self.client is not None
        # State may have changed while the request was queued; re-validate
        # against the snapshot before touching any connection.
        fingerprint = self.follow_fingerprint
        if fingerprint is None:
            fingerprint = self.client._follow_fingerprint(self.request)
        active_scope, record = _scope_validate_request(
            self.client,
            self.request,
            fingerprint,
            scope=self.follow_scope,
        )
        assert active_scope is not None
        scope = active_scope
        self._follow_streaming = (
            self.request.body_producer is not None
            or self.request.streaming_callback is not None
        )
        scope._request_active = True
        scope._streaming = self._follow_streaming
        if record is not None:
            self.follow_record = record
            record.retain()  # released in _follow_cleanup
            record.busy = True
            record.streaming = self._follow_streaming
        try:
            while True:
                completed = await self._follow_one_hop()
                if completed:
                    return
                if not self._follow_redirect_pending:
                    break
                # Next redirect hop on the same stream.
                self._follow_redirect_pending = False
        except BaseException:
            self._follow_cleanup(dead=True)
            raise

    async def _follow_one_hop(self) -> bool:
        """Sends one request and reads its response on the pinned stream.

        Returns True if the request has already been completed via a timeout
        callback (in which case the caller should stop).
        """
        assert self.follow_scope is not None
        assert self.client is not None
        self.parsed = urllib.parse.urlsplit(_unicode(self.request.url))
        if self.parsed.scheme not in ("http", "https"):
            raise ValueError("Unsupported url scheme: %s" % self.request.url)
        # urlsplit results have hostname and port results, but they
        # didn't support ipv6 literals until python 2.7.
        netloc = self.parsed.netloc
        if "@" in netloc:
            userpass, _, netloc = netloc.rpartition("@")
        host, port = httputil.split_host_and_port(netloc)
        if port is None:
            port = 443 if self.parsed.scheme == "https" else 80
        if re.match(r"^\[.*\]$", host):
            # raw ipv6 addresses in urls are enclosed in brackets
            host = host[1:-1]
        self.parsed_hostname = host

        hop_fingerprint = self.client._follow_fingerprint(self.request)
        record = self.follow_record
        if record is not None:
            # Every redirect hop must stay on the pinned connection. The
            # target host/port may change (the request is simply written to
            # the same socket), but anything that would require a different
            # kind of connection (scheme, proxy, TLS settings) cannot.
            if record.dead or record.payload.get("stream") is None:
                raise HTTPClientConnectionFollowError(
                    "the pinned connection is closed or no longer usable"
                )
            # Layout: (scheme, host, port, <proxy and TLS settings...>).
            if (
                record.impl_class is not type(self.client)
                or record.fingerprint[0] != hop_fingerprint[0]
                or record.fingerprint[3:] != hop_fingerprint[3:]
            ):
                raise HTTPClientConnectionFollowError(
                    "the redirect would leave the pinned connection"
                )
            stream = record.payload["stream"]
            if stream.closed():
                record.mark_dead()
                raise HTTPClientConnectionFollowError(
                    "the pinned connection was closed"
                )
            self.stream = stream
            self.stream.set_close_callback(self.on_connection_close)
        else:
            if self.request.allow_ipv6 is False:
                af = socket.AF_INET
            else:
                af = socket.AF_UNSPEC
            ssl_options = self._get_ssl_options(self.parsed.scheme)
            source_ip = None
            if self.request.network_interface:
                if is_valid_ip(self.request.network_interface):
                    source_ip = self.request.network_interface
                else:
                    raise ValueError(
                        "Unrecognized IPv4 or IPv6 address for network_interface, got %r"
                        % (self.request.network_interface,)
                    )
            if self.request.connect_timeout and self.request.request_timeout:
                connect_timeout = min(
                    self.request.connect_timeout, self.request.request_timeout
                )
            elif self.request.connect_timeout:
                connect_timeout = self.request.connect_timeout
            elif self.request.request_timeout:
                connect_timeout = self.request.request_timeout
            else:
                connect_timeout = 0
            try:
                stream = await self.tcp_client.connect(
                    host,
                    port,
                    af=af,
                    ssl_options=ssl_options,
                    max_buffer_size=self.max_buffer_size,
                    source_ip=source_ip,
                    timeout=(
                        self.start_time + connect_timeout - self.io_loop.time()
                        if connect_timeout
                        else None
                    ),
                )
            except TimeoutError as e:
                # TCPClient's timeout messages describe which phase of
                # the connection timed out (DNS, TCP, or TLS).
                raise HTTPTimeoutError(str(e) or "Timeout while connecting") from None

            if self.final_callback is None:
                # final_callback is cleared if we've hit our timeout.
                stream.close()
                self._follow_cleanup(dead=True)
                return True
            self.stream = stream
            self.stream.set_close_callback(self.on_connection_close)

        if self.final_callback is None:
            self._follow_cleanup(dead=True)
            return True
        if self.request.request_timeout:
            self._timeout = self.io_loop.add_timeout(
                self.start_time + self.request.request_timeout,
                functools.partial(self._on_timeout, "during request"),
            )
        if (
            self.request.method not in self._SUPPORTED_METHODS
            and not self.request.allow_nonstandard_methods
        ):
            raise KeyError("unknown method %s" % self.request.method)
        for key in (
            "proxy_host",
            "proxy_port",
            "proxy_username",
            "proxy_password",
            "proxy_auth_mode",
        ):
            if getattr(self.request, key, None):
                raise NotImplementedError("%s not supported" % key)
        # Keep the connection alive; do not force "Connection: close".
        if self.request.headers.get("Connection", "").lower() == "close":
            self.request.headers["Connection"] = "keep-alive"
        req_path = self._prepare_request_message()
        self.connection = self._create_connection(stream)
        start_line = httputil.RequestStartLine(self.request.method, req_path, "")
        self.connection.write_headers(start_line, self.request.headers)
        if self.request.expect_100_continue:
            await self.connection.read_response(self)
        else:
            await self._write_body(True)
        # finish() (called from read_response) either scheduled the next
        # redirect hop or finalized the request and delivered the response.
        pending_error = getattr(self, "_follow_pending_error", None)
        if pending_error is not None:
            self._follow_pending_error = None
            self._follow_cleanup(dead=True)
            if hasattr(self, "stream"):
                self.stream.close()
            raise pending_error
        if self._follow_redirect_pending:
            return False
        # Final response: either park the live stream in the scope or close
        # it when the peer asked for (or signaled) closure.
        reusable = getattr(self, "_follow_final_reusable", False)
        if reusable and hasattr(self, "stream"):
            # _read_message() cleared the stream's close callback on its way
            # out, so notice an asynchronous close while idle.
            self.stream.set_close_callback(self._follow_idle_mark_dead)
            self._follow_cleanup(dead=False)
        else:
            self._follow_cleanup(dead=True)
            if hasattr(self, "stream"):
                self.stream.close()
        return True

    def _follow_bind_after_send(self) -> None:
        """Binds the freshly used connection to the scope after the request
        has been sent successfully.

        A new scope with no binding of its own acquires the connection it
        just used; an inner scope that borrowed an enclosing scope's
        connection binds the same record itself.
        """
        scope = self.follow_scope
        if scope is None or scope._record is not None:
            return
        if self.follow_record is None:
            assert self.client is not None
            fingerprint = self.client._follow_fingerprint(self.request)
            record = _FollowedConnection(
                type(self.client),
                fingerprint,
                {"stream": self.stream},
                self._follow_release_record,
            )
            record.retain()  # released in _follow_cleanup
            self.follow_record = record
            record.busy = True
            record.streaming = self._follow_streaming
        record = self.follow_record
        scope._bind(record)

    def _follow_response_reusable(self) -> bool:
        """Whether the response just read leaves the connection ready for
        another request."""
        if self.stream is None or self.stream.closed():
            return False
        headers = self.headers
        if headers is None or self._response_version is None:
            return False
        connection_header = headers.get("Connection", "").lower()
        if self._response_version == "HTTP/1.1":
            if connection_header == "close":
                return False
        elif connection_header != "keep-alive":
            return False
        # The body must be self-delimited; otherwise it ran until the
        # server closed the connection.
        if self.request.method == "HEAD" or (self.code in (204, 304)):
            return True
        if "Content-Length" in headers:
            return True
        te = headers.get("Transfer-Encoding", "")
        if "chunked" in te.lower():
            return True
        return False

    def _follow_idle_mark_dead(self) -> None:
        record = self.follow_record
        if record is not None:
            record.mark_dead()

    def _follow_cleanup(self, dead: bool) -> None:
        if self._follow_cleaned:
            return
        self._follow_cleaned = True
        record = self.follow_record
        scope = self.follow_scope
        self._remove_timeout()
        if scope is not None:
            scope._request_active = False
            scope._streaming = False
        if record is not None:
            if dead:
                record.mark_dead()
            record.busy = False
            record.streaming = False
            try:
                self.stream.set_close_callback(None)
            except Exception:
                pass
            record.release()

    def _get_ssl_options(self, scheme: str) -> None | dict[str, Any] | ssl.SSLContext:
        if scheme == "https":
            if self.request.ssl_options is not None:
                return self.request.ssl_options
            # If we are using the defaults, don't construct a
            # new SSLContext.
            if (
                self.request.validate_cert
                and self.request.ca_certs is None
                and self.request.client_cert is None
                and self.request.client_key is None
            ):
                return _client_ssl_defaults()
            ssl_ctx = ssl.create_default_context(
                ssl.Purpose.SERVER_AUTH, cafile=self.request.ca_certs
            )
            if not self.request.validate_cert:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
            if self.request.client_cert is not None:
                ssl_ctx.load_cert_chain(
                    self.request.client_cert, self.request.client_key
                )
            if hasattr(ssl, "OP_NO_COMPRESSION"):
                # See netutil.ssl_options_to_context
                ssl_ctx.options |= ssl.OP_NO_COMPRESSION
            return ssl_ctx
        return None

    def _on_timeout(self, info: str | None = None) -> None:
        """Timeout callback of _HTTPConnection instance.

        Raise a `HTTPTimeoutError` when a timeout occurs.

        :info string key: More detailed timeout information.
        """
        self._timeout = None
        error_message = f"Timeout {info}" if info else "Timeout"
        if self.final_callback is not None:
            self._handle_exception(
                HTTPTimeoutError, HTTPTimeoutError(error_message), None
            )

    def _remove_timeout(self) -> None:
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None

    def _create_connection(self, stream: IOStream) -> HTTP1Connection:
        stream.set_nodelay(True)
        connection = HTTP1Connection(
            stream,
            True,
            HTTP1ConnectionParameters(
                # Inside a follow scope the stream is kept alive and bound
                # to the scope; elsewhere the historical "one request per
                # connection" behavior is unchanged.
                no_keep_alive=self.follow_scope is None,
                max_header_size=self.max_header_size,
                max_body_size=self.max_body_size,
                decompress=bool(self.request.decompress_response),
            ),
            self._sockaddr,
        )
        return connection

    async def _write_body(self, start_read: bool) -> None:
        assert self.connection is not None
        if self.request.body is not None:
            self.connection.write(self.request.body)
        elif self.request.body_producer is not None:
            fut = self.request.body_producer(self.connection.write)
            if fut is not None:
                await fut
        self.connection.finish()
        if self.follow_scope is not None:
            # The full request has been sent successfully; now the
            # connection may be pinned to the scope.
            self._follow_bind_after_send()
        if start_read:
            try:
                await self.connection.read_response(self)
            except StreamClosedError:
                if not self._handle_exception(*sys.exc_info()):
                    raise

    def _release(self) -> None:
        if self.release_callback is not None:
            release_callback = self.release_callback
            self.release_callback = None  # type: ignore
            release_callback()

    def _run_callback(self, response: HTTPResponse) -> None:
        self._release()
        if self.final_callback is not None:
            final_callback = self.final_callback
            self.final_callback = None  # type: ignore
            self.io_loop.add_callback(final_callback, response)

    def _handle_exception(
        self,
        typ: "Optional[Type[BaseException]]",
        value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if self.final_callback is not None:
            self._remove_timeout()
            if isinstance(value, StreamClosedError):
                if value.real_error is None:
                    value = HTTPStreamClosedError("Stream closed")
                else:
                    value = value.real_error
            # A failure tears down the pinned connection; mark it dead so
            # later requests in the scope fail instead of reconnecting.
            if self.follow_scope is not None:
                self._follow_cleanup(dead=True)
            self._run_callback(
                HTTPResponse(
                    self.request,
                    599,
                    error=value,
                    request_time=self.io_loop.time() - self.start_time,
                    start_time=self.start_wall_time,
                )
            )

            if hasattr(self, "stream"):
                # TODO: this may cause a StreamClosedError to be raised
                # by the connection's Future.  Should we cancel the
                # connection more gracefully?
                self.stream.close()
            return True
        else:
            # If our callback has already been called, we are probably
            # catching an exception that is not caused by us but rather
            # some child of our callback. Rather than drop it on the floor,
            # pass it along, unless it's just the stream being closed.
            return isinstance(value, StreamClosedError)

    def on_connection_close(self) -> None:
        if self.final_callback is not None:
            if self.stream.error:
                # This is called from the connection's read loop, which
                # only logs exceptions, so the error must go to
                # _handle_exception instead of being raised.
                error = self.stream.error
                self._handle_exception(type(error), error, error.__traceback__)
                return
            try:
                raise HTTPStreamClosedError("Connection closed")
            except HTTPStreamClosedError:
                self._handle_exception(*sys.exc_info())

    async def headers_received(
        self,
        first_line: httputil.ResponseStartLine | httputil.RequestStartLine,
        headers: httputil.HTTPHeaders,
    ) -> None:
        assert isinstance(first_line, httputil.ResponseStartLine)
        self._response_version = first_line.version
        if self.request.expect_100_continue and first_line.code == 100:
            await self._write_body(False)
            return
        self.code = first_line.code
        self.reason = first_line.reason
        self.headers = headers

        if self._should_follow_redirect():
            return

        if self.request.header_callback is not None:
            # Reassemble the start line.
            self.request.header_callback("%s %s %s\r\n" % first_line)
            for k, v in self.headers.get_all():
                self.request.header_callback(f"{k}: {v}\r\n")
            self.request.header_callback("\r\n")

    def _should_follow_redirect(self) -> bool:
        if self.request.follow_redirects:
            assert self.request.max_redirects is not None
            return (
                self.code in (301, 302, 303, 307, 308)
                and self.request.max_redirects > 0
                and self.headers is not None
                and self.headers.get("Location") is not None
            )
        return False

    def finish(self) -> None:
        assert self.code is not None
        data = bytes(self.chunks)
        self._remove_timeout()
        original_request = getattr(self.request, "original_request", self.request)
        if self._should_follow_redirect():
            assert isinstance(self.request, _RequestProxy)
            assert self.headers is not None
            new_request = copy.copy(self.request.request)
            new_request.url = urllib.parse.urljoin(
                self.request.url, self.headers["Location"]
            )
            new_request.headers = self.request.headers.copy()
            parsed_orig_url = urllib.parse.urlsplit(original_request.url)
            parsed_new_url = urllib.parse.urlsplit(new_request.url)
            if (
                parsed_orig_url.scheme != parsed_new_url.scheme
                or parsed_orig_url.netloc != parsed_new_url.netloc
            ):
                # Cross-origin redirect: strip auth headers.
                # Note that while there is no formal specification of headers that should be
                # stripped here, libcurl strips the Authorization and Cookie headers, so we
                # do the same.
                # Reference:
                # https://github.com/curl/curl/blob/01d8191b25a05e8fa91553a6c0d48acb99907d26/lib/http.c#L1827-L1828
                #
                # Note that checking for cross-origin redirects is a crude heuristic. It is both
                # too weak (e.g. cookies that have a path attribute may need to be stripped even on
                # same-origin redirects) and too strong (e.g. cookies may be kept on cross-host
                # redirects within the same domain). However, we cannot know the full details of
                # the cookie policy at this layer, so we use the same heuristic as libcurl.
                # Applications that need more control over behavior on redirects can set
                # follow_redirects=False and handle 3xx responses themselves.
                new_request.auth_username = None
                new_request.auth_password = None
                if "@" in parsed_new_url.netloc:
                    if parsed_new_url.port is not None:
                        new_netloc = f"{parsed_new_url.hostname}:{parsed_new_url.port}"
                    else:
                        assert parsed_new_url.hostname is not None
                        new_netloc = parsed_new_url.hostname
                    parsed_new_url = parsed_new_url._replace(netloc=new_netloc)
                new_request.url = urllib.parse.urlunsplit(parsed_new_url)
                for h in ["Authorization", "Cookie"]:
                    try:
                        del new_request.headers[h]
                    except KeyError:
                        pass
            assert self.request.max_redirects is not None
            new_request.max_redirects = self.request.max_redirects - 1
            del new_request.headers["Host"]
            # https://tools.ietf.org/html/rfc7231#section-6.4
            #
            # The original HTTP spec said that after a 301 or 302
            # redirect, the request method should be preserved.
            # However, browsers implemented this by changing the
            # method to GET, and the behavior stuck. 303 redirects
            # always specified this POST-to-GET behavior, arguably
            # for *all* methods, but libcurl < 7.70 only does this
            # for POST, while libcurl >= 7.70 does it for other methods.
            if (self.code == 303 and self.request.method != "HEAD") or (
                self.code in (301, 302) and self.request.method == "POST"
            ):
                new_request.method = "GET"
                new_request.body = None  # type: ignore
                for h in [
                    "Content-Length",
                    "Content-Type",
                    "Content-Encoding",
                    "Transfer-Encoding",
                ]:
                    try:
                        del new_request.headers[h]
                    except KeyError:
                        pass
            new_request.original_request = original_request  # type: ignore
            if self.follow_scope is not None:
                # Every redirect hop must reuse the pinned connection, so
                # drive the next hop ourselves instead of going through
                # client.fetch() (which would open a new connection).
                assert self.client is not None
                if not self._follow_response_reusable():
                    # Raised from the runner after read_response returns;
                    # raising here would be swallowed by the read loop.
                    self._follow_pending_error = HTTPClientConnectionFollowError(
                        "cannot follow the redirect on the pinned connection"
                    )
                    return
                old_proxy = self.request
                self.request = cast(
                    HTTPRequest, _RequestProxy(new_request, old_proxy.defaults)
                )
                # Reset per-response delegate state for the next hop.
                self.code = None
                self.headers = None
                self.chunks = bytearray()
                self._response_version = None
                self.connection = None
                self._follow_redirect_pending = True
                return
            final_callback = self.final_callback
            self.final_callback = None  # type: ignore
            self._release()
            assert self.client is not None
            fut = self.client.fetch(new_request, raise_error=False)
            fut.add_done_callback(lambda f: final_callback(f.result()))
            self._on_end_request()
            return
        if self.request.streaming_callback:
            buffer = BytesIO()
        else:
            buffer = BytesIO(data)  # TODO: don't require one big string?
        response = HTTPResponse(
            original_request,
            self.code,
            reason=getattr(self, "reason", None),
            headers=self.headers,
            request_time=self.io_loop.time() - self.start_time,
            start_time=self.start_wall_time,
            buffer=buffer,
            effective_url=self.request.url,
        )
        if self.follow_scope is not None:
            # Keep the stream open (handed back to the runner, which either
            # parks it in the scope or closes it); just deliver the response.
            self._follow_final_reusable = self._follow_response_reusable()
            self._run_callback(response)
            return
        self._run_callback(response)
        self._on_end_request()

    def _on_end_request(self) -> None:
        self.stream.close()

    def data_received(self, chunk: bytes) -> Awaitable[None] | None:
        if self._should_follow_redirect():
            # We're going to follow a redirect so just discard the body.
            return None
        if self.request.streaming_callback is not None:
            return self.request.streaming_callback(chunk)
        else:
            self.chunks += chunk
            return None


if __name__ == "__main__":
    AsyncHTTPClient.configure(SimpleAsyncHTTPClient)
    main()
