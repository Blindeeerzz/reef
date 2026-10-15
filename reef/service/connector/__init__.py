"""Opt-in outbound connection from a Reef service to its owner's console, optionally starting that service."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
import webbrowser
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from reef.service.connector.http_policy import HttpPolicy
from reef.service.connector.http_transport import NativeHttpConnector
from reef.service.connector.runtime import HTTPFailure, JSONClient, endpoint_url
from reef.service.connector.service import ReefService, address_in_use, serve_address
from reef.service.connector.state import ConnectionConfig, ConnectorState


async def authorize(config: ConnectionConfig, state: ConnectorState, *, no_browser: bool) -> ConnectionConfig:
    async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=True) as session:
        platform = JSONClient(session, config["platform_url"], config.get("connector_token", ""))
        if platform.token:
            return config
        pairing = await platform.request(
            "/api/connector/pair",
            body={
                "protocol": 2,
                "instance_id": config["instance_id"],
                "name": config["name"],
            },
        )
        verification = pairing.get("verification_uri")
        device_token, user_code = pairing.get("device_token"), pairing.get("user_code")
        expires_in = pairing.get("expires_in", 600)
        if (
            not isinstance(verification, str)
            or not verification
            or not isinstance(device_token, str)
            or not device_token
            or not isinstance(user_code, str)
            or not user_code
        ):
            raise ValueError("Pairing requires a login URL, device token and user code")
        if not isinstance(expires_in, (int, float)) or isinstance(expires_in, bool) or not 0 < expires_in <= 600:
            raise ValueError("Invalid pairing expiry")
        login_url, platform_url = urlsplit(endpoint_url(verification)), urlsplit(config["platform_url"])
        if (login_url.scheme, login_url.netloc) != (platform_url.scheme, platform_url.netloc):
            raise ValueError("Platform returned a login URL on a different origin")
        platform.token = device_token
        print(
            f"Open {verification}\nDevice code: {user_code}\nPaste this code into the page to connect {config['name']}.",
            flush=True,
        )
        print("HTTP forwarding uses your local policy. Approved responses leave this machine.", flush=True)
        if not no_browser:
            webbrowser.open(verification)
        deadline = time.monotonic() + expires_in
        while time.monotonic() < deadline:
            status = await platform.request("/api/connector/pair")
            if status.get("status") == "approved":
                config["connector_token"] = platform.token
                runtime_id = status.get("runtime_id")
                if not isinstance(runtime_id, str) or not runtime_id:
                    raise ValueError("Pairing requires a runtime ID")
                config["runtime_id"] = runtime_id
                state.save(config)
                return config
            await asyncio.sleep(3)
        raise RuntimeError("Connection code expired; run reef connect again")


async def run_connector(config: ConnectionConfig, state: ConnectorState) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop.set)
    async with (
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar(), trust_env=True) as platform_session,
        aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as local_session,
    ):
        serve = config.get("serve")
        service = (
            ReefService(
                serve["arguments"],
                config["reef_url"],
                config.get("reef_token", ""),
                Path(serve["directory"]),
                state.directory / "serve.log",
            )
            if serve
            else None
        )
        connector = NativeHttpConnector(
            JSONClient(platform_session, config["platform_url"], config["connector_token"]),
            JSONClient(local_session, config["reef_url"], config.get("reef_token", "")),
            state,
            HttpPolicy.from_mapping(config.get("http_policy")),
            config["instance_id"],
            service,
        )
        await connector.run(stop)


async def check_reef(config: ConnectionConfig) -> str | None:
    """Return a local readiness warning without publishing scenario summaries."""
    url = config["reef_url"]
    async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as session:
        try:
            await JSONClient(session, url, config.get("reef_token", "")).request("/reef/scenarios", timeout=5)
        except aiohttp.ClientConnectorError:
            return f"Nothing answered at {url}. Start Reef there or use its serving address."
        except HTTPFailure as exc:
            return f"Reef at {url} returned HTTP {exc.status}. Check its address and service token."
        except (aiohttp.ClientError, TimeoutError):
            return f"Reef at {url} did not answer. Check its address."
    return None


def connector_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reef connect",
        description="Connect a Reef runtime to your API platform account, optionally starting it with --serve.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8900", help="Reef service URL")
    parser.add_argument("--platform", default="https://api.reefinfra.ai", help="API platform URL")
    parser.add_argument("--name", help="Name shown in the console")
    parser.add_argument(
        "--reef-token-env", default="REEF_TOKEN", help="Environment variable holding the local service token"
    )
    parser.add_argument("--state-dir", type=Path, help="Private state directory for this connection")
    parser.add_argument("--foreground", action="store_true", help="Stay in the foreground (for service supervisors)")
    parser.add_argument("--no-browser", action="store_true", help="Print the login link without opening a browser")
    parser.add_argument(
        "--http-policy", type=Path, help="Required local JSON authorization policy; saved for subsequent starts"
    )
    parser.add_argument(
        "--stop", action="store_true", help="Stop this connector, keeping its credentials for a later restart"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show this connector's local process and state directory"
    )
    parser.add_argument(
        "--serve",
        nargs=argparse.REMAINDER,
        metavar="SERVE_OPTION",
        help="Start `reef serve` with the options that follow on the --url address, and stop it with the connector."
        " Put it last.",
    )
    parser.add_argument("--reauthorize", action="store_true", help="Pair again after revoking a connection")
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    return parser


def connection_running(state: ConnectorState) -> bool:
    try:
        with state.lock():
            return False
    except RuntimeError:
        return True


def main(argv: list[str] | None = None) -> None:
    """Authorize once, then maintain the connection independently of the browser."""
    args = connector_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        platform_url, reef_url = endpoint_url(args.platform), endpoint_url(args.url)
        connection_key = hashlib.sha256(f"{platform_url}\n{reef_url}".encode()).hexdigest()[:16]
        directory = args.state_dir or Path.home() / ".reef" / "connections" / connection_key
        state = ConnectorState(directory)
        try:
            running = connection_running(state)
            config = state.load()
            serving = bool(config.get("serve"))
            if args.status:
                print(f"Connector {'running' if running else 'stopped'}\nState: {directory}")
                if serving:
                    print(f"Reef logs: {directory / 'serve.log'}")
                return
            if args.stop:
                if running:
                    os.kill(int((directory / "connector.pid").read_text()), signal.SIGTERM)
                if serving:
                    print("Connector stopping, together with the Reef service it started.")
                else:
                    print(
                        "Connector stopping. Reef continues serving; revoke access in the console to remove"
                        " authorization."
                    )
                return
            if running:
                raise RuntimeError("This connector is already running; use --status or --stop")
            if args.run:
                if not config.get("connector_token"):
                    raise RuntimeError("Run reef connect to authorize this instance first")
                with state.lock():
                    (directory / "connector.pid").write_text(str(os.getpid()))
                    asyncio.run(run_connector(config, state))
                return
            if config and (config.get("platform_url") != platform_url or config.get("reef_url") != reef_url):
                raise ValueError(
                    "This state directory belongs to another endpoint; use its original --platform and --url"
                )
            if args.serve is not None and address_in_use(*serve_address(reef_url, args.serve)):
                raise RuntimeError(f"A service already answers at {reef_url}. Stop it, or connect without --serve.")
            policy = config.get("http_policy")
            if args.http_policy is not None:
                policy = json.loads(args.http_policy.read_text(encoding="utf-8"))
            HttpPolicy.from_mapping(policy)
            if args.reauthorize:
                config.pop("connector_token", None)
                config.pop("runtime_id", None)
            config.update(
                {
                    "http_policy": policy,
                    "platform_url": platform_url,
                    "reef_url": reef_url,
                    "instance_id": config.get("instance_id") or str(uuid.uuid4()),
                    "name": args.name or config.get("name") or socket.gethostname(),
                    "reef_token": os.environ.get(args.reef_token_env, config.get("reef_token", "")),
                    # Each invocation decides whether the connector starts Reef; serve runs where connect ran.
                    "serve": (
                        {"arguments": args.serve, "directory": str(Path.cwd())} if args.serve is not None else None
                    ),
                }
            )
            # Save identity before authorization so interrupted logins cannot create new identities.
            state.save(config)
            if args.serve is None:
                warning = asyncio.run(check_reef(config))
                if warning is not None:
                    print(
                        f"Warning: {warning}\nThe connection still completes; the console shows the runtime"
                        " once Reef answers. To start Reef with the connector, add --serve and its options.",
                        file=sys.stderr,
                    )
            config = asyncio.run(authorize(config, state, no_browser=args.no_browser))
            if args.serve is not None:
                print(f"Starting Reef at {reef_url}. Logs: {directory / 'serve.log'}", flush=True)
            if args.foreground:
                with state.lock():
                    (directory / "connector.pid").write_text(str(os.getpid()))
                    # Only a connector that holds the lock runs, so the line waits for it.
                    print(f"Connected {config['name']}. Open {platform_url}/local", flush=True)
                    asyncio.run(run_connector(config, state))
            else:
                log_path = directory / "connector.log"
                descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(descriptor, "a") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "reef.service.connector", "--run", "--state-dir", str(directory)],
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=output,
                        start_new_session=True,
                    )
                time.sleep(0.5)
                if process.poll() is not None:
                    raise RuntimeError(f"Connector did not start. See {log_path}")
                print(f"Connected {config['name']}. Open {platform_url}/local\nLogs: {log_path}")
        finally:
            state.close()
    except (RuntimeError, ValueError, OSError, aiohttp.ClientError) as exc:
        print(f"reef connect: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
