"""Bounded JSON requests for connector pairing and local readiness checks."""

from __future__ import annotations

import ipaddress
import json
from typing import cast
from urllib.parse import urlsplit

import aiohttp


class HTTPFailure(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def endpoint_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost" or hostname.endswith(".localhost")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("Use HTTPS, or HTTP on localhost")
    if not hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint URLs cannot contain credentials, a query, or a fragment")
    _ = parsed.port
    return value.rstrip("/")


class JSONClient:
    """HTTP boundary with bounded responses, no cookies, redirects, or implicit retries."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, token: str = ""):
        self.session = session
        self.base_url = endpoint_url(base_url)
        self.token = token

    async def request(
        self,
        path: str,
        *,
        body: dict[str, object] | None = None,
        timeout: float = 15,
    ) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with self.session.request(
            "GET" if body is None else "POST",
            self.base_url + path,
            headers=headers,
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=False,
        ) as response:
            chunks = bytearray()
            async for chunk in response.content.iter_chunked(8192):
                chunks.extend(chunk)
                if len(chunks) > 2 * 1024 * 1024:
                    raise HTTPFailure(502, "Response exceeds the connector size limit")
            if response.status < 200 or response.status >= 300:
                # Do not upload runtime error bodies: they can contain credentials, paths, or provider payloads.
                raise HTTPFailure(response.status, f"HTTP {response.status}")
            try:
                result = json.loads(chunks)
            except (ValueError, UnicodeDecodeError) as exc:
                raise HTTPFailure(502, "Expected a JSON object") from exc
            if not isinstance(result, dict):
                raise HTTPFailure(502, "Expected a JSON object")
            return cast(dict[str, object], result)
