"""Protocol 2: bounded, locally authorized native HTTP forwarding."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from typing import cast
from urllib.parse import urljoin, urlsplit

import aiohttp
from yarl import URL

from reef.service.connector.http_policy import HttpPolicy
from reef.service.connector.runtime import JSONClient
from reef.service.connector.service import ReefService
from reef.service.connector.state import ConnectorState

logger = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
RESPONSE_CHUNK_BYTES = 32 * 1024
REQUEST_HEADERS = frozenset(
    (
        "accept",
        "content-type",
        "anthropic-version",
        "anthropic-beta",
        "x-request-id",
        "x-reef-scenario",
        "x-reef-release-id",
        "traceparent",
        "tracestate",
    )
)
RESPONSE_HEADERS = frozenset(
    (
        "content-type",
        "cache-control",
        "content-disposition",
        "retry-after",
        "x-request-id",
        "x-reef-agent-record-id",
        "x-reef-release-id",
        "x-reef-artifact-version",
    )
)


def message_object(data: str) -> dict[str, object]:
    value: object = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Protocol messages must be JSON objects")
    return cast(dict[str, object], value)


def request_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Request ID must be a UUID")
    return str(uuid.UUID(value))


def filtered_headers(headers: Iterable[tuple[str, str]], permitted: frozenset[str]) -> list[tuple[str, str]]:
    items = list(headers)
    connection_headers = {
        token.strip().lower() for name, value in items if name.lower() == "connection" for token in value.split(",")
    }
    return [(name, value) for name, value in items if name.lower() in permitted - connection_headers]


@dataclass(frozen=True)
class HttpRequest:
    id: str
    method: str
    path: str
    query: str
    headers: Mapping[str, str]
    body: bytes
    scenario: str | None

    @classmethod
    def from_message(cls, message: Mapping[str, object], policy: HttpPolicy) -> HttpRequest:
        if set(message) - {"type", "id", "method", "path", "query", "headers", "body"}:
            raise ValueError("Unknown request fields")
        identity = request_id(message.get("id"))
        method, path, query = message.get("method"), message.get("path"), message.get("query", "")
        if method not in ("GET", "POST", "DELETE") or not isinstance(method, str):
            raise ValueError("Unsupported HTTP method")
        if not isinstance(path, str) or len(path) > 2048 or not isinstance(query, str) or len(query) > 8192:
            raise ValueError("Invalid HTTP path or query")
        configured_headers = message.get("headers", {})
        if not isinstance(configured_headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in configured_headers.items()
        ):
            raise ValueError("Headers must contain string names and values")
        headers: dict[str, str] = {}
        for key, value in cast(dict[str, str], configured_headers).items():
            name = key.lower()
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9a-z-]+", name) or name in headers:
                raise ValueError("Invalid or repeated header name")
            if any(
                ord(character) < 32 or ord(character) == 127 or 0xD800 <= ord(character) <= 0xDFFF
                for character in value
            ):
                raise ValueError("Invalid header value")
            headers[name] = value
        encoded_body = message.get("body", "")
        if not isinstance(encoded_body, str):
            raise ValueError("Body must be base64 text")
        try:
            body = base64.b64decode(encoded_body, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 body") from exc
        if len(body) > policy.max_request_bytes:
            raise ValueError("Request body exceeds the local limit")
        if body and method != "POST":
            raise ValueError("Only POST requests can carry a body")
        scenario = policy.authorize(method, path, query, headers, body)
        return cls(identity, method, path, query, headers, body, scenario)

    def fingerprint(self) -> str:
        encoded = json.dumps(
            {
                "method": self.method,
                "path": self.path,
                "query": self.query,
                "headers": dict(self.headers),
                "body": base64.b64encode(self.body).decode("ascii"),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class NativeHttpConnector:
    """Forward approved requests; connection loss never repeats a write."""

    def __init__(
        self,
        platform: JSONClient,
        local: JSONClient,
        state: ConnectorState,
        policy: HttpPolicy,
        instance_id: str,
        service: ReefService | None = None,
    ):
        self.platform = platform
        self.local = local
        self.state = state
        self.policy = policy
        self.instance_id = instance_id
        self.service = service
        self.send_lock = asyncio.Lock()
        self.limits = {
            "max_concurrent_requests": policy.max_concurrent_requests,
            "max_request_bytes": policy.max_request_bytes,
            "max_message_bytes": MAX_MESSAGE_BYTES,
            "response_chunk_bytes": RESPONSE_CHUNK_BYTES,
            "request_timeout_seconds": policy.request_timeout_seconds,
        }

    async def send(self, websocket: aiohttp.ClientWebSocketResponse, message: Mapping[str, object]) -> None:
        try:
            async with asyncio.timeout(15):
                async with self.send_lock:
                    await websocket.send_json(dict(message))
        except (aiohttp.ClientError, TimeoutError, ConnectionError) as exc:
            await websocket.close()
            raise ConnectionError("Forwarding connection closed") from exc

    async def report_error(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        identity: str | None,
        code: str,
        outcome: str = "not_dispatched",
    ) -> None:
        with suppress(ConnectionError):
            await self.send(websocket, {"type": "error", "id": identity, "code": code, "outcome": outcome})

    async def forward(self, websocket: aiohttp.ClientWebSocketResponse, request: HttpRequest) -> None:
        outcome = "unknown"
        headers = dict(filtered_headers(request.headers.items(), REQUEST_HEADERS))
        if self.local.token:
            headers["authorization"] = f"Bearer {self.local.token}"
        if request.scenario is not None:
            headers["x-reef-scenario"] = request.scenario
        headers["accept-encoding"] = "identity"
        address = self.local.base_url + request.path + ("?" + request.query if request.query else "")
        try:
            async with asyncio.timeout(self.policy.request_timeout_seconds):
                async with self.local.session.request(
                    request.method,
                    URL(address, encoded=True),
                    headers=headers,
                    data=request.body or None,
                    allow_redirects=False,
                    auto_decompress=False,
                    timeout=aiohttp.ClientTimeout(total=self.policy.request_timeout_seconds),
                ) as response:
                    location = response.headers.get("location")
                    if location:
                        target, origin = urlsplit(urljoin(address, location)), urlsplit(self.local.base_url)
                        if (target.scheme, target.netloc) != (origin.scheme, origin.netloc):
                            await self.report_error(websocket, request.id, "external_redirect", "unknown")
                            return
                    if response.headers.get("content-encoding", "identity") != "identity":
                        await self.report_error(websocket, request.id, "unsupported_encoding", "unknown")
                        return
                    await self.send(
                        websocket,
                        {
                            "type": "response",
                            "id": request.id,
                            "status": response.status,
                            "headers": filtered_headers(response.headers.items(), RESPONSE_HEADERS),
                        },
                    )
                    async for chunk in response.content.iter_chunked(RESPONSE_CHUNK_BYTES):
                        await self.send(
                            websocket,
                            {"type": "chunk", "id": request.id, "body": base64.b64encode(chunk).decode("ascii")},
                        )
                    await self.send(websocket, {"type": "end", "id": request.id})
                    outcome = "completed"
        except asyncio.CancelledError:
            await self.report_error(websocket, request.id, "cancelled", "unknown")
            raise
        except TimeoutError:
            await self.report_error(websocket, request.id, "deadline_exceeded", "unknown")
        except (aiohttp.ClientError, ConnectionError):
            await self.report_error(websocket, request.id, "connection_interrupted", "unknown")
        finally:
            if request.method != "GET":
                self.state.finish_http_request(request.id, outcome)
            logger.info("Forwarded request %s: %s", request.id, outcome)

    async def heartbeat(self, websocket: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            await asyncio.sleep(15)
            await self.send(websocket, {"type": "heartbeat", "protocol": 2})

    async def run_session(self, websocket: aiohttp.ClientWebSocketResponse, stop: asyncio.Event) -> None:
        await self.send(
            websocket,
            {
                "type": "hello",
                "protocol": 2,
                "instance_id": self.instance_id,
                "runtime_version": version("reef-infra"),
                "limits": self.limits,
                "capabilities": {
                    "record_bodies": self.policy.allow_record_bodies,
                    "inference": self.policy.allow_inference,
                },
            },
        )
        ready_message = await websocket.receive(timeout=15)
        if ready_message.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError("Receiver did not confirm native HTTP protocol 2")
        try:
            ready = message_object(ready_message.data)
        except ValueError as exc:
            raise RuntimeError("Receiver must confirm native HTTP protocol 2") from exc
        agreed = ready.get("limits")
        if ready.get("type") != "ready" or ready.get("protocol") != 2 or agreed != self.limits:
            raise RuntimeError("Receiver must confirm native HTTP protocol 2 and its limits")
        if not isinstance(agreed, dict) or any(type(value) is not int for value in agreed.values()):
            raise RuntimeError("Receiver returned invalid protocol limits")
        active: dict[str, asyncio.Task[None]] = {}
        heartbeat = asyncio.create_task(self.heartbeat(websocket))
        stop_wait = asyncio.create_task(stop.wait())
        receive = asyncio.create_task(websocket.receive())
        try:
            while not stop.is_set():
                done, _ = await asyncio.wait((receive, stop_wait, heartbeat), return_when=asyncio.FIRST_COMPLETED)
                if stop_wait in done:
                    break
                if heartbeat in done:
                    await heartbeat
                message = receive.result()
                if message.type != aiohttp.WSMsgType.TEXT:
                    break
                receive = asyncio.create_task(websocket.receive())
                for finished_id in list(active):
                    if active[finished_id].done():
                        finished_task = active.pop(finished_id)
                        with suppress(asyncio.CancelledError):
                            await finished_task
                identity: str | None = None
                try:
                    frame = message_object(message.data)
                    identity = request_id(frame.get("id"))
                    if frame.get("type") == "cancel":
                        cancelled_task = active.get(identity)
                        if cancelled_task is not None:
                            cancelled_task.cancel()
                        else:
                            await self.report_error(websocket, identity, "not_active", "unknown")
                        continue
                    if frame.get("type") != "request":
                        raise ValueError("Expected request or cancel")
                    request = HttpRequest.from_message(frame, self.policy)
                    prior = self.state.http_request(request.id)
                    if prior is not None:
                        code = "already_dispatched" if prior.fingerprint == request.fingerprint() else "id_conflict"
                        await self.report_error(websocket, identity, code, prior.outcome)
                        continue
                    if request.id in active:
                        await self.report_error(websocket, identity, "already_active", "unknown")
                        continue
                    if len(active) >= self.policy.max_concurrent_requests:
                        await self.report_error(websocket, identity, "capacity_exceeded")
                        continue
                    if request.method != "GET":
                        self.state.start_http_request(request.id, request.fingerprint())
                    active[request.id] = asyncio.create_task(self.forward(websocket, request))
                except (ValueError, UnicodeError):
                    await self.report_error(websocket, identity, "request_rejected")
        finally:
            receive.cancel()
            heartbeat.cancel()
            stop_wait.cancel()
            for task in active.values():
                task.cancel()
            results = await asyncio.gather(receive, heartbeat, stop_wait, *active.values(), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, (asyncio.CancelledError, ConnectionError)
                ):
                    raise result

    async def run(self, stop: asyncio.Event) -> None:
        async def reject_redirect(
            session: aiohttp.ClientSession, context: object, parameters: aiohttp.TraceRequestRedirectParams
        ) -> None:
            raise RuntimeError("The forwarding endpoint must not redirect")

        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(reject_redirect)
        if self.service is not None:
            self.service.start()
        try:
            async with aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar(),
                trust_env=True,
                trace_configs=[trace],
                timeout=aiohttp.ClientTimeout(total=15),
            ) as outbound:
                endpoint = URL(self.platform.base_url + "/api/connector/stream")
                endpoint = endpoint.with_scheme("wss" if endpoint.scheme == "https" else "ws")
                retry_seconds = 1
                while not stop.is_set():
                    try:
                        async with outbound.ws_connect(
                            endpoint,
                            headers={"authorization": f"Bearer {self.platform.token}"},
                            max_msg_size=MAX_MESSAGE_BYTES,
                            heartbeat=30,
                            timeout=aiohttp.ClientWSTimeout(ws_close=5),
                        ) as websocket:
                            await self.run_session(websocket, stop)
                            retry_seconds = 1
                    except aiohttp.WSServerHandshakeError as exc:
                        if exc.status in (401, 403):
                            raise RuntimeError("Connection revoked; run reef connect --reauthorize") from exc
                        if 400 <= exc.status < 500:
                            raise RuntimeError("Receiver does not support native HTTP protocol 2") from exc
                        logger.warning("Forwarding connection unavailable; reconnecting")
                    except (aiohttp.ClientError, TimeoutError, ConnectionError, ValueError):
                        logger.warning("Forwarding connection interrupted; reconnecting")
                    if not stop.is_set():
                        try:
                            await asyncio.wait_for(stop.wait(), timeout=retry_seconds)
                        except TimeoutError:
                            retry_seconds = min(30, retry_seconds * 2)
        finally:
            if self.service is not None:
                await self.service.stop()
