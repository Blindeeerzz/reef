"""Local authorization and resource limits for native HTTP forwarding."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit


@dataclass(frozen=True)
class HttpRoute:
    method: str
    path: str

    def matches(self, method: str, path: str, scenario: str | None) -> bool:
        if self.method != method:
            return False
        template = self.path.replace("{scenario}", scenario or "")
        if self.path.endswith("/*"):
            return path.startswith(template[:-1])
        return path == template


@dataclass(frozen=True)
class HttpPolicy:
    scenarios: tuple[str, ...]
    routes: tuple[HttpRoute, ...]
    allow_record_bodies: bool = False
    allow_inference: bool = False
    max_concurrent_requests: int = 4
    max_request_bytes: int = 1024 * 1024
    request_timeout_seconds: int = 600

    @classmethod
    def from_mapping(cls, value: object) -> HttpPolicy:
        if not isinstance(value, dict):
            raise ValueError("HTTP policy must be a JSON object")
        permitted = {
            "scenarios",
            "routes",
            "allow_record_bodies",
            "allow_inference",
            "max_concurrent_requests",
            "max_request_bytes",
            "request_timeout_seconds",
        }
        if set(value) - permitted:
            raise ValueError("HTTP policy contains unknown settings")
        scenarios = value.get("scenarios")
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("HTTP policy requires a non-empty scenarios list")
        for scenario in scenarios:
            if (
                not isinstance(scenario, str)
                or not scenario
                or scenario.strip() != scenario
                or scenario in (".", "..")
                or re.search(r"[/\\\x00-\x1f\x7f%?#]", scenario)
            ):
                raise ValueError("HTTP policy contains an invalid scenario")
        if "*" in scenarios and scenarios != ["*"]:
            raise ValueError('The all-scenarios grant must be written as ["*"]')
        configured = value.get("routes")
        if not isinstance(configured, list) or not configured:
            raise ValueError("HTTP policy requires a non-empty routes list")
        routes = []
        for rule in configured:
            if not isinstance(rule, dict) or set(rule) != {"method", "path"}:
                raise ValueError("Each HTTP route requires method and path")
            method, path = rule["method"], rule["path"]
            if method not in ("GET", "POST", "DELETE") or not isinstance(path, str):
                raise ValueError("HTTP routes support GET, POST and DELETE")
            literal = path.replace("{scenario}", "scenario")
            if literal.endswith("/*"):
                literal = literal[:-1] + "route"
            literal = validate_http_path(literal)
            if any(character in literal for character in "*{}"):
                raise ValueError("Route patterns support {scenario} and a final /* only")
            routes.append(HttpRoute(method, unquote(path, errors="strict")))
        flags = {}
        for name in ("allow_record_bodies", "allow_inference"):
            flag = value.get(name, False)
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a boolean")
            flags[name] = flag
        limits = {}
        for name, default, maximum in (
            ("max_concurrent_requests", 4, 16),
            ("max_request_bytes", 1024 * 1024, 1024 * 1024),
            ("request_timeout_seconds", 600, 3600),
        ):
            limit = value.get(name, default)
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")
            limits[name] = limit
        return cls(
            tuple(scenarios),
            tuple(routes),
            flags["allow_record_bodies"],
            flags["allow_inference"],
            limits["max_concurrent_requests"],
            limits["max_request_bytes"],
            limits["request_timeout_seconds"],
        )

    def authorize(self, method: str, path: str, query: str, headers: Mapping[str, str], body: bytes) -> str | None:
        decoded = validate_http_path(path)
        if (
            query.startswith("?")
            or "#" in query
            or any(ord(character) <= 32 or ord(character) >= 127 for character in query)
            or re.search(r"%(?![0-9a-fA-F]{2})", query)
        ):
            raise ValueError("Invalid HTTP query")
        selectors = []
        segments = decoded.split("/")
        if len(segments) > 3 and segments[1:3] == ["reef", "scenarios"]:
            selectors.append(segments[3])
        if headers.get("x-reef-scenario"):
            selectors.append(headers["x-reef-scenario"])
        parameters = parse_qs(query, keep_blank_values=True)
        selectors.extend(parameters.get("scenario", ()))
        if body and ("json" in headers.get("content-type", "") or body.lstrip().startswith(b"{")):
            try:
                payload: object = json.loads(body)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError("Invalid JSON request body") from exc
            if isinstance(payload, dict):
                if "scenario" in payload:
                    selectors.append(payload["scenario"])
                if decoded == "/reef/scenarios" and method == "POST" and "name" in payload:
                    selectors.append(payload["name"])
        if (
            any(
                not isinstance(item, str)
                or not item
                or item.strip() != item
                or re.search(r"[/\\\x00-\x1f\x7f%?#\ud800-\udfff]", item)
                for item in selectors
            )
            or len(set(selectors)) > 1
        ):
            raise ValueError("Scenario selectors disagree")
        scenario = selectors[0] if selectors else None
        if self.scenarios != ("*",) and (scenario is None or scenario not in self.scenarios):
            raise ValueError("Scenario is not authorized")
        if self.scenarios != ("*",) and decoded in ("/reef/status", "/reef/scenarios") and method == "GET":
            raise ValueError("Global discovery requires the all-scenarios grant")
        if not any(route.matches(method, decoded, scenario) for route in self.routes):
            raise ValueError("HTTP route is not authorized")
        if not self.allow_record_bodies and any(
            segment in ("records", "requests", "trajectories") for segment in segments
        ):
            raise ValueError("Record bodies require an explicit local grant")
        if (decoded.startswith("/v1/") or "/evaluation/v1/" in decoded) and not self.allow_inference:
            raise ValueError("Inference requires an explicit local grant")
        return scenario


def validate_http_path(path: str) -> str:
    """Accept relative native API paths without alternate targets or ambiguous escapes."""
    if any(ord(character) <= 32 or ord(character) >= 127 for character in path):
        raise ValueError("HTTP path must use URL-encoded characters")
    if not path.startswith(("/reef/", "/v1/")) or path.startswith("//"):
        raise ValueError("Only relative /reef/ and /v1/ routes can be forwarded")
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("HTTP path must not contain an address, query or fragment")
    if re.search(r"%(?![0-9a-fA-F]{2})", path):
        raise ValueError("Invalid path escape")
    decoded = unquote(path, errors="strict")
    if re.search(r"[\\\x00-\x1f\x7f%?#]", decoded):
        raise ValueError("Ambiguous HTTP path")
    if re.search(r"%(?:2f|5c)", path, re.IGNORECASE) or any(
        part in ("", ".", "..") for part in decoded.split("/")[1:]
    ):
        raise ValueError("HTTP path contains an empty segment or escape")
    return decoded
