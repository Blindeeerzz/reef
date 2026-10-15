"""Native connector contracts against real local HTTP and WebSocket peers."""

import asyncio
import base64
import json
import os
import signal
import socket
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.cli import main
from reef.service.connector import authorize, connection_running
from reef.service.connector import main as connect_main
from reef.service.connector.http_policy import HttpPolicy
from reef.service.connector.http_transport import HttpRequest, NativeHttpConnector
from reef.service.connector.runtime import HTTPFailure, JSONClient, endpoint_url
from reef.service.connector.service import ReefService, serve_address
from reef.service.connector.state import ConnectorState


def native_policy(**changes):
    value = {
        "scenarios": ["demo"],
        "routes": [
            {"method": "GET", "path": "/reef/scenarios/{scenario}/*"},
            {"method": "POST", "path": "/reef/scenarios/{scenario}/*"},
            {"method": "POST", "path": "/v1/chat/completions"},
            {"method": "POST", "path": "/reef/train"},
        ],
    }
    value.update(changes)
    return HttpPolicy.from_mapping(value)


def write_policy(directory):
    path = directory / "policy.json"
    path.write_text(json.dumps({"scenarios": ["*"], "routes": [{"method": "GET", "path": "/reef/*"}]}))
    return path


def request_frame(path, *, method="GET", body=b"", headers=None, query="", identity=None):
    return {
        "type": "request",
        "id": identity or str(uuid.uuid4()),
        "method": method,
        "path": path,
        "query": query,
        "headers": headers or {},
        "body": base64.b64encode(body).decode(),
    }


class NativeReceiver:
    def __init__(self, protocol: int = 2, limit_updates: dict[str, object] | None = None):
        self.protocol = protocol
        self.limit_updates = limit_updates or {}
        self.frames: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.connected = asyncio.Event()
        self.websocket: web.WebSocketResponse | None = None
        self.transport: asyncio.Transport | None = None
        self.hello: dict[str, object] = {}
        self.authorization = ""

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.authorization = request.headers.get("authorization", "")
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        self.websocket = websocket
        self.transport = request.transport
        self.hello = await websocket.receive_json()
        await websocket.send_json(
            {"type": "ready", "protocol": self.protocol, "limits": {**self.hello["limits"], **self.limit_updates}}
        )
        self.connected.set()
        async for message in websocket:
            if message.type == aiohttp.WSMsgType.TEXT:
                await self.frames.put(json.loads(message.data))
        return websocket

    async def send(self, frame):
        assert self.websocket is not None
        await self.websocket.send_json(frame)

    async def result(self, identity):
        frames = []
        async with asyncio.timeout(5):
            while True:
                frame = await self.frames.get()
                if frame.get("id") != identity:
                    continue
                frames.append(frame)
                if frame["type"] in ("end", "error"):
                    return frames


@asynccontextmanager
async def native_connection(tmp_path, app, policy, receiver=None):
    receiver = receiver if receiver is not None else NativeReceiver()
    remote = web.Application()
    remote.router.add_get("/api/connector/stream", receiver.handle)
    async with (
        TestServer(app) as local_server,
        TestServer(remote) as remote_server,
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as platform_session,
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as local_session,
    ):
        state = ConnectorState(tmp_path)
        connector = NativeHttpConnector(
            JSONClient(platform_session, str(remote_server.make_url("/")).rstrip("/"), "platform-credential"),
            JSONClient(local_session, str(local_server.make_url("/")).rstrip("/"), "local-credential"),
            state,
            policy,
            "test-instance",
        )
        stop = asyncio.Event()
        task = asyncio.create_task(connector.run(stop))
        try:
            await asyncio.wait_for(receiver.connected.wait(), 5)
            yield receiver, state, connector, task
        finally:
            stop.set()
            try:
                await asyncio.wait_for(task, 5)
            finally:
                state.close()


@pytest.mark.parametrize("status", [200, 400, 409, 500])
def test_native_responses_match_direct_api_fields_and_errors(tmp_path, status):
    async def run():
        seen = []

        async def handler(request):
            seen.append((request.method, request.query_string, dict(request.headers), await request.read()))
            return web.json_response(
                {"model": {"name": "example"}, "next_cursor": "next", "details": [1, 2]},
                status=status,
                headers={"x-request-id": "native-id", "set-cookie": "private=value", "connection": "close"},
            )

        app = web.Application()
        app.router.add_post("/reef/scenarios/demo/update", handler)
        async with native_connection(tmp_path, app, native_policy()) as (receiver, state, connector, *_):
            payload = b'{"model":{"model":"example","url":"https://provider.example/v1"}}'
            frame = request_frame(
                "/reef/scenarios/demo/update",
                method="POST",
                body=payload,
                query="limit=2&after=next",
                headers={
                    "content-type": "application/json",
                    "authorization": "Bearer remote-secret",
                    "cookie": "remote-cookie=value",
                    "x-reef-scenario": "demo",
                },
            )
            await receiver.send(frame)
            frames = await receiver.result(frame["id"])
            assert frames[0]["status"] == status
            headers = dict(frames[0]["headers"])
            assert headers["x-request-id"] == "native-id"
            assert "set-cookie" not in headers and "connection" not in headers
            body = b"".join(base64.b64decode(part["body"]) for part in frames if part["type"] == "chunk")
            direct = await connector.local.session.post(
                connector.local.base_url + frame["path"],
                data=payload,
            )
            async with direct:
                assert direct.status == status
                assert body == await direct.read()
            method, query, forwarded, sent = seen[0]
            assert method == "POST" and query == "limit=2&after=next" and sent == payload
            assert forwarded["Authorization"] == "Bearer local-credential"
            assert forwarded["x-reef-scenario"] == "demo" and "Cookie" not in forwarded
            assert receiver.authorization == "Bearer platform-credential"
            assert "local-credential" not in json.dumps([*frames, receiver.hello])
            assert state.http_request(frame["id"]).outcome == "completed"

    asyncio.run(run())


@pytest.mark.parametrize(
    "frame,changes",
    [
        (request_frame("https://other.example/reef/scenarios/demo/update"), {}),
        (request_frame("//other.example/reef/scenarios/demo/update"), {}),
        (request_frame("/reef/scenarios/demo/../other"), {}),
        (request_frame("/reef/scenarios/demo/%2e%2e/other"), {}),
        (request_frame("/reef/scenarios/demo%2fother/releases"), {}),
        (request_frame("/reef/scenarios/demo/evaluation/v1/chat/completions", method="POST"), {}),
        (request_frame("/reef/scenarios/demo/releases", query="scenario=demo&scenario=other"), {}),
        (request_frame("/reef/scenarios/demo/releases", query="scenario=demo%20"), {}),
        (request_frame("/reef/scenarios/demo/releases", query="scenario=%other"), {}),
        (
            request_frame("/reef/train", method="POST", body=b'{"scenario":"demo\\r\\nx: secret"}'),
            {"scenarios": ["*"]},
        ),
        (request_frame("/reef/scenarios/demo/releases", headers={"accept": "\ud800"}), {}),
        (request_frame("/reef/scenarios/de mo/releases"), {}),
        (request_frame("/reef/scenarios/other/releases"), {}),
        (request_frame("/reef/scenarios/demo/releases", query="scenario=other"), {}),
        (request_frame("/reef/scenarios/demo/releases", headers={"x-reef-scenario": "other"}), {}),
        (request_frame("/reef/scenarios/demo/update", method="POST", body=b'{"scenario":"other"}'), {}),
        (request_frame("/reef/scenarios/demo/records"), {}),
        (request_frame("/v1/chat/completions", method="POST", headers={"x-reef-scenario": "demo"}), {}),
        (request_frame("/reef/scenarios/demo/update", method="POST", body=b"12345"), {"max_request_bytes": 4}),
        (request_frame("/reef/scenarios/demo/update", method="POST", body=b"{"), {}),
    ],
)
def test_native_policy_rejects_unauthorized_or_ambiguous_calls(frame, changes):
    with pytest.raises(ValueError):
        HttpRequest.from_message(frame, native_policy(**changes))


def test_native_policy_requires_grants_for_global_discovery_and_private_bodies():
    policy = native_policy(routes=[{"method": "GET", "path": "/reef/*"}])
    with pytest.raises(ValueError, match="Global discovery"):
        HttpRequest.from_message(request_frame("/reef/scenarios", headers={"x-reef-scenario": "demo"}), policy)
    whole = replace(policy, scenarios=("*",))
    assert HttpRequest.from_message(request_frame("/reef/scenarios"), whole).scenario is None
    record = request_frame("/reef/scenarios/demo/records")
    assert HttpRequest.from_message(record, replace(policy, allow_record_bodies=True)).scenario == "demo"


def test_policy_matches_encoded_scenario_paths():
    policy = native_policy(
        scenarios=["demo one"], routes=[{"method": "GET", "path": "/reef/scenarios/demo%20one/releases"}]
    )
    request = HttpRequest.from_message(request_frame("/reef/scenarios/demo%20one/releases"), policy)
    assert request.scenario == "demo one"


@pytest.mark.parametrize("change", [{"allow_inference": "yes"}, {"max_concurrent_requests": True}, {"unknown": 1}])
def test_native_policy_rejects_invalid_settings(change):
    with pytest.raises(ValueError):
        native_policy(**change)


def test_native_streams_preserve_bytes_and_arrive_before_completion(tmp_path):
    async def run():
        finish = asyncio.Event()
        first = b'data: {"delta":"first"}\n\n'
        second = b'data: {"delta":"second"}\n\ndata: [DONE]\n\n'

        async def handler(request):
            response = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await response.prepare(request)
            await response.write(first)
            await finish.wait()
            await response.write(second)
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        async with native_connection(tmp_path, app, native_policy(allow_inference=True)) as (receiver, *_):
            frame = request_frame("/v1/chat/completions", method="POST", headers={"x-reef-scenario": "demo"})
            await receiver.send(frame)
            start = await asyncio.wait_for(receiver.frames.get(), 3)
            chunk = await asyncio.wait_for(receiver.frames.get(), 3)
            assert start["type"] == "response" and chunk["type"] == "chunk"
            assert base64.b64decode(chunk["body"]) == first
            finish.set()
            tail = await receiver.result(frame["id"])
            assert b"".join(base64.b64decode(part["body"]) for part in tail if part["type"] == "chunk") == second

    asyncio.run(run())


def test_native_cancellation_and_capacity_do_not_start_extra_work(tmp_path):
    async def run():
        finish = asyncio.Event()
        calls = []

        async def handler(request):
            calls.append(request.path)
            response = web.StreamResponse()
            await response.prepare(request)
            await finish.wait()
            return response

        app = web.Application()
        app.router.add_post("/reef/scenarios/demo/update", handler)
        try:
            async with native_connection(tmp_path, app, native_policy(max_concurrent_requests=1)) as (
                receiver,
                state,
                *_,
            ):
                first = request_frame("/reef/scenarios/demo/update", method="POST")
                await receiver.send(first)
                assert (await asyncio.wait_for(receiver.frames.get(), 3))["type"] == "response"
                second = request_frame("/reef/scenarios/demo/update", method="POST")
                await receiver.send(second)
                assert (await receiver.result(second["id"]))[-1]["code"] == "capacity_exceeded"
                await receiver.send({"type": "cancel", "id": first["id"]})
                result = await receiver.result(first["id"])
                assert result[-1]["code"] == "cancelled" and result[-1]["outcome"] == "unknown"
                assert len(calls) == 1 and state.http_request(second["id"]) is None
                finish.set()
        finally:
            finish.set()

    asyncio.run(run())


def test_native_write_ids_survive_restart_without_replay_or_payload_storage(tmp_path):
    async def run():
        calls = []

        async def handler(request):
            calls.append(await request.read())
            return web.json_response({"accepted": True})

        app = web.Application()
        app.router.add_post("/reef/scenarios/demo/update", handler)
        frame = request_frame("/reef/scenarios/demo/update", method="POST", body=b'{"model":"private-body"}')
        async with native_connection(tmp_path, app, native_policy()) as (receiver, *_):
            await receiver.send(frame)
            assert (await receiver.result(frame["id"]))[-1]["type"] == "end"
        app2 = web.Application()
        app2.router.add_post("/reef/scenarios/demo/update", handler)
        async with native_connection(tmp_path, app2, native_policy()) as (receiver, *_):
            await receiver.send(frame)
            assert (await receiver.result(frame["id"]))[-1]["code"] == "already_dispatched"
            changed = {**frame, "body": base64.b64encode(b'{"model":"changed"}').decode()}
            await receiver.send(changed)
            assert (await receiver.result(frame["id"]))[-1]["code"] == "id_conflict"
        assert len(calls) == 1
        assert b"private-body" not in (tmp_path / "requests.sqlite3").read_bytes()

    asyncio.run(run())


def test_native_disconnect_after_dispatch_stays_unknown(tmp_path):
    async def run():
        finish = asyncio.Event()

        async def handler(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await finish.wait()
            return response

        app = web.Application()
        app.router.add_post("/reef/scenarios/demo/update", handler)
        frame = request_frame("/reef/scenarios/demo/update", method="POST")
        try:
            async with native_connection(tmp_path, app, native_policy()) as connection:
                receiver, state, task = connection[0], connection[1], connection[3]
                await receiver.send(frame)
                assert (await asyncio.wait_for(receiver.frames.get(), 3))["type"] == "response"
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert state.http_request(frame["id"]).outcome == "unknown"
                finish.set()
        except asyncio.CancelledError:
            pass
        finally:
            finish.set()
        reopened = ConnectorState(tmp_path)
        try:
            assert reopened.http_request(frame["id"]).outcome == "unknown"
        finally:
            reopened.close()

    asyncio.run(run())


def test_native_http_requires_policy_before_pairing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        connect_main(["--state-dir", str(tmp_path)])
    assert "HTTP policy" in capsys.readouterr().err


@pytest.mark.parametrize(
    "protocol,limits", [(1, {}), (2, {"max_request_bytes": 1}), (2, {"max_concurrent_requests": 4.0})]
)
def test_native_receiver_must_confirm_protocol(tmp_path, protocol, limits):
    async def run():
        with pytest.raises(RuntimeError, match=r"confirm native HTTP|invalid protocol limits"):
            async with native_connection(
                tmp_path, web.Application(), native_policy(), NativeReceiver(protocol, limits)
            ):
                pass

    asyncio.run(run())


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://user:secret@example.com",
        "https://example.com?token=secret",
        "file:///tmp/a",
        "http://127.0.0.1:bad",
        "https://example.com/#secret",
    ],
)
def test_endpoint_rejects_insecure_or_credential_urls(url):
    with pytest.raises(ValueError):
        endpoint_url(url)


def test_http_boundary_keeps_tokens_separate_and_blocks_redirects():
    async def run():
        seen = []

        async def handler(request):
            seen.append((request.path, request.headers.get("Authorization"), request.headers.get("Cookie")))
            if request.path == "/redirect":
                raise web.HTTPFound("/secret")
            if request.path == "/invalid":
                return web.Response(text="accepted, but not JSON")
            if request.path == "/large":
                return web.Response(body=b"x" * (2 * 1024 * 1024 + 1))
            return web.json_response({"ok": True}, headers={"Set-Cookie": "private=value"})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server, aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as session:
            platform = JSONClient(session, str(server.make_url("")).rstrip("/"), "platform-test-token")
            local = JSONClient(session, platform.base_url, "local-test-token")
            await platform.request("/cloud", body={"ready": True})
            await local.request("/reef/status")
            for path in ("/redirect", "/invalid", "/large"):
                with pytest.raises(HTTPFailure):
                    await local.request(path)
            assert seen[:2] == [
                ("/cloud", "Bearer platform-test-token", None),
                ("/reef/status", "Bearer local-test-token", None),
            ]
            assert not any(path == "/secret" for path, _, _ in seen)

    asyncio.run(run())


def test_pairing_saves_only_after_approval_and_reuses_identity(tmp_path, monkeypatch, capsys):
    async def run():
        state = ConnectorState(tmp_path)
        calls = []

        async def handler(request):
            calls.append((request.method, request.headers.get("Authorization")))
            if request.method == "POST":
                body = await request.json()
                assert body == {"protocol": 2, "instance_id": "stable-id", "name": "my-machine"}
                return web.json_response(
                    {
                        "verification_uri": str(server.make_url("/local/authorize")),
                        "device_token": "platform-test-token",
                        "user_code": "ABCD-EF12-3456",
                        "expires_in": 30,
                    }
                )
            return web.json_response({"status": "approved", "runtime_id": "runtime-id"})

        app = web.Application()
        app.router.add_route("*", "/api/connector/pair", handler)
        try:
            async with TestServer(app) as server:
                config = {
                    "platform_url": str(server.make_url("")).rstrip("/"),
                    "instance_id": "stable-id",
                    "name": "my-machine",
                    "reef_token": "local-test-token",
                }
                await authorize(config, state, no_browser=True)
            assert state.load()["connector_token"] == "platform-test-token"
            assert calls == [("POST", None), ("GET", "Bearer platform-test-token")]
            output = capsys.readouterr().out
            assert "test-token" not in output
            assert "Device code: ABCD-EF12-3456" in output
            assert "Paste this code into the page" in output
            assert "/local/authorize" in output
            assert "/local/connect/ABCD" not in output
        finally:
            state.close()

    asyncio.run(run())


def test_connect_cli_is_dispatched_without_starting_a_deployment(capsys):
    with pytest.raises(SystemExit) as result:
        main(["connect", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--foreground" in output and "--reef-token-env" in output and "--stop" in output


@pytest.mark.parametrize(
    ("url", "arguments"),
    [
        ("https://127.0.0.1:8901", []),
        ("http://127.0.0.1", []),
        ("http://127.0.0.1:8901/reef", []),
        ("http://127.0.0.1:8901", ["--reef.port", "9000"]),
        ("http://127.0.0.1:8901", ["--reef.host=0.0.0.0"]),
    ],
)
def test_serve_address_comes_only_from_the_connector_url(url, arguments):
    with pytest.raises(ValueError):
        serve_address(url, arguments)


def test_serve_address_accepts_a_loopback_port():
    assert serve_address("http://localhost:8901", ["--recipe", "reefine"]) == ("localhost", 8901)


def test_connect_warns_before_pairing_when_reef_does_not_answer(tmp_path, monkeypatch, capsys):
    async def stop_before_pairing(*_args, **_kwargs):
        raise RuntimeError("pairing skipped")

    monkeypatch.setattr("reef.service.connector.authorize", stop_before_pairing)
    url = f"http://127.0.0.1:{unused_port()}"
    with pytest.raises(SystemExit):
        main(
            [
                "connect",
                "--http-policy",
                str(write_policy(tmp_path)),
                "--url",
                url,
                "--platform",
                "https://platform.example",
                "--state-dir",
                str(tmp_path),
            ]
        )
    error = capsys.readouterr().err
    assert f"Warning: Nothing answered at {url}" in error and "--serve" in error


def test_connect_serve_refuses_an_address_already_in_use(tmp_path, capsys):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        with pytest.raises(SystemExit):
            main(["connect", "--url", url, "--state-dir", str(tmp_path), "--serve", "--model", "ollama/test"])
    assert f"A service already answers at {url}" in capsys.readouterr().err


def test_serve_reports_its_exit_code(tmp_path):
    async def run():
        service = ReefService(
            ["--recipe", "missing-profile"], "http://127.0.0.1:8901", "", tmp_path, tmp_path / "serve.log"
        )
        service.start()
        await asyncio.to_thread(service.process.wait, 30)
        return service.process.returncode

    assert asyncio.run(run()) == 2
    assert "missing-profile" in (tmp_path / "serve.log").read_text()


class CliReceiver(NativeReceiver):
    def __init__(self):
        super().__init__()
        self.pair_count = 0
        self.revoked = False

    async def pair(self, request):
        if request.method == "POST":
            self.pair_count += 1
            payload = await request.json()
            assert payload["protocol"] == 2
            return web.json_response(
                {
                    "verification_uri": str(request.url.with_path("/local/authorize")),
                    "device_token": "platform-credential",
                    "user_code": "ABCD-EF12-3456",
                    "expires_in": 60,
                }
            )
        return web.json_response({"status": "approved", "runtime_id": "runtime-id"})

    async def handle(self, request):
        if self.revoked:
            return web.Response(status=401)
        return await super().handle(request)

    def application(self):
        app = web.Application()
        app.router.add_route("*", "/api/connector/pair", self.pair)
        app.router.add_get("/api/connector/stream", self.handle)
        return app


async def cli_process(directory, platform_url, reef_url, *options):
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "reef.cli",
        "connect",
        "--no-browser",
        "--platform",
        platform_url,
        "--url",
        reef_url,
        "--state-dir",
        str(directory),
        *options,
        cwd=directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def wait_for_connection_state(state, expected):
    async with asyncio.timeout(15):
        while connection_running(state) is not expected:
            await asyncio.sleep(0.05)


@pytest.mark.skipif(os.name == "nt", reason="POSIX background process lifecycle")
def test_cli_reuses_saved_policy_and_identity_and_reauthorizes_after_revocation(tmp_path):
    async def run():
        receiver = CliReceiver()
        state = ConnectorState(tmp_path)
        async with TestServer(receiver.application()) as server:
            address = str(server.make_url("/")).rstrip("/")

            async def cli(*options):
                process = await cli_process(tmp_path, address, address, *options)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                assert process.returncode == 0, stderr.decode()
                return stdout.decode()

            try:
                assert "Connected" in await cli("--http-policy", str(write_policy(tmp_path)))
                await asyncio.wait_for(receiver.connected.wait(), 5)
                identity = state.load()["instance_id"]
                assert "running" in await cli("--status")
                await cli("--stop")
                await wait_for_connection_state(state, False)
                receiver.connected.clear()
                await cli()
                await asyncio.wait_for(receiver.connected.wait(), 5)
                assert state.load()["instance_id"] == identity and receiver.pair_count == 1
                receiver.revoked = True
                await receiver.websocket.close()
                await wait_for_connection_state(state, False)
                log = (tmp_path / "connector.log").read_text()
                assert "--reauthorize" in log and "platform-credential" not in log
                receiver.revoked = False
                receiver.connected.clear()
                await cli("--reauthorize")
                await asyncio.wait_for(receiver.connected.wait(), 5)
                assert receiver.pair_count == 2 and state.load()["instance_id"] == identity
            finally:
                if connection_running(state):
                    await cli("--stop")
                    await wait_for_connection_state(state, False)
                state.close()

    asyncio.run(run())


@pytest.mark.skipif(os.name == "nt", reason="POSIX process lifecycle")
@pytest.mark.parametrize("foreground", [True, False])
def test_cli_starts_and_stops_reef_and_forwards_its_native_status(tmp_path, foreground):
    async def run():
        receiver = CliReceiver()
        state = ConnectorState(tmp_path)
        reef_url = f"http://127.0.0.1:{unused_port()}"
        options = ["--foreground"] if foreground else []
        async with TestServer(receiver.application()) as server:
            address = str(server.make_url("/")).rstrip("/")
            process = await cli_process(
                tmp_path,
                address,
                reef_url,
                "--http-policy",
                str(write_policy(tmp_path)),
                *options,
                "--serve",
                "--model",
                "ollama/test",
            )
            try:
                await asyncio.wait_for(receiver.connected.wait(), 15)
                async with aiohttp.ClientSession() as session, asyncio.timeout(30):
                    client = JSONClient(session, reef_url)
                    while True:
                        try:
                            direct = await client.request("/reef/status")
                            break
                        except aiohttp.ClientConnectorError:
                            await asyncio.sleep(0.1)
                    frame = request_frame("/reef/status")
                    await receiver.send(frame)
                    frames = await receiver.result(frame["id"])
                    body = b"".join(base64.b64decode(part["body"]) for part in frames if part["type"] == "chunk")
                    assert frames[0]["status"] == 200 and json.loads(body).keys() == direct.keys()
                assert (tmp_path / "serve.log").exists()
            finally:
                if foreground and process.returncode is None:
                    process.send_signal(signal.SIGTERM)
                else:
                    await asyncio.wait_for(process.communicate(), 15)
                    stop_process = await cli_process(tmp_path, address, reef_url, "--stop")
                    await asyncio.wait_for(stop_process.communicate(), 15)
                _, stderr = await asyncio.wait_for(process.communicate(), 15)
                await wait_for_connection_state(state, False)
                state.close()
            assert process.returncode == 0, stderr.decode()
            async with aiohttp.ClientSession() as session:
                with pytest.raises(aiohttp.ClientConnectorError):
                    await JSONClient(session, reef_url).request("/reef/status")

    asyncio.run(run())


def test_private_configuration_and_process_lock(tmp_path):
    state = ConnectorState(tmp_path)
    try:
        state.save({"instance_id": "instance", "connector_token": "private-test-token"})
        assert state.load()["instance_id"] == "instance"
        with state.lock(), pytest.raises(RuntimeError, match="already running"), state.lock():
            pass
        if os.name != "nt":
            assert tmp_path.stat().st_mode & 0o777 == 0o700
            assert state.config_path.stat().st_mode & 0o777 == 0o600
            assert (tmp_path / "requests.sqlite3").stat().st_mode & 0o777 == 0o600
    finally:
        state.close()


def test_native_deadline_does_not_repeat_dispatched_work(tmp_path):
    async def run():
        finish = asyncio.Event()
        calls = []

        async def handler(request):
            calls.append(request.path)
            await finish.wait()
            return web.Response()

        app = web.Application()
        app.router.add_post("/reef/scenarios/demo/update", handler)
        try:
            async with native_connection(tmp_path, app, native_policy(request_timeout_seconds=1)) as (
                receiver,
                state,
                *_,
            ):
                frame = request_frame("/reef/scenarios/demo/update", method="POST")
                await receiver.send(frame)
                result = await receiver.result(frame["id"])
                assert result[-1]["code"] == "deadline_exceeded" and result[-1]["outcome"] == "unknown"
                await receiver.send(frame)
                assert (await receiver.result(frame["id"]))[-1]["code"] == "already_dispatched"
                assert len(calls) == 1 and state.http_request(frame["id"]).outcome == "unknown"
                finish.set()
        finally:
            finish.set()

    asyncio.run(run())


def test_native_redirects_do_not_reach_another_origin(tmp_path):
    async def run():
        reached = []

        async def secret(request):
            reached.append(request.headers.get("authorization"))
            return web.Response()

        target = web.Application()
        target.router.add_get("/secret", secret)
        async with TestServer(target) as target_server:

            async def redirect(request):
                raise web.HTTPFound(str(target_server.make_url("/secret")))

            app = web.Application()
            app.router.add_get("/reef/scenarios/demo/releases", redirect)
            async with native_connection(tmp_path, app, native_policy()) as (receiver, *_):
                frame = request_frame("/reef/scenarios/demo/releases")
                await receiver.send(frame)
                assert (await receiver.result(frame["id"]))[-1]["code"] == "external_redirect"
            remote = web.Application()
            remote.router.add_get("/api/connector/stream", redirect)
            async with TestServer(remote) as remote_server, aiohttp.ClientSession() as session:
                state = ConnectorState(tmp_path)
                connector = NativeHttpConnector(
                    JSONClient(session, str(remote_server.make_url("/")).rstrip("/"), "platform-credential"),
                    JSONClient(session, "http://127.0.0.1:8900", "local-credential"),
                    state,
                    native_policy(),
                    "test-instance",
                )
                try:
                    with pytest.raises(RuntimeError, match="must not redirect"):
                        await connector.run(asyncio.Event())
                finally:
                    state.close()
            assert reached == []

    asyncio.run(run())


def test_slow_receiver_limits_the_local_response_reader(tmp_path):
    async def run():
        started = asyncio.Event()
        written_chunks = 0
        chunk = b"x" * 32768
        total_chunks = 2048

        async def handler(request):
            nonlocal written_chunks
            response = web.StreamResponse()
            await response.prepare(request)
            started.set()
            for _ in range(total_chunks):
                await response.write(chunk)
                written_chunks += 1
            return response

        app = web.Application()
        app.router.add_get("/reef/scenarios/demo/releases", handler)
        async with native_connection(tmp_path, app, native_policy()) as (receiver, *_):
            frame = request_frame("/reef/scenarios/demo/releases")
            await receiver.send(frame)
            await asyncio.wait_for(started.wait(), 3)
            transport = receiver.transport
            assert transport is not None
            # Stop consuming peer messages at the server transport, like a stalled receiver.
            transport.pause_reading()
            try:
                await asyncio.sleep(0.2)
                assert written_chunks < total_chunks
            finally:
                transport.resume_reading()
            received_bytes = 0
            async with asyncio.timeout(15):
                while True:
                    part = await receiver.frames.get()
                    if part["type"] == "chunk":
                        decoded = base64.b64decode(part["body"])
                        assert len(decoded) <= 32768
                        received_bytes += len(decoded)
                    if part["type"] == "end":
                        break
            assert received_bytes == total_chunks * len(chunk)

    asyncio.run(run())
