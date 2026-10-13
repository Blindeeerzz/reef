"""Tasks prepared for Harbor's E2B environment: the image's environment read from its registry and stated in a copy
of the task, which the checks and the plays run while the written task stays as it is."""

from __future__ import annotations

import base64
import dataclasses
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.core.tasks import read_harbor_task, write_harbor_task
from reef.harness.client.tasks import EpisodeRow, TaskLab
from reef.record2task.generator import OracleUnavailable, ReefTaskPlays, harbor_task, oracle_check
from reef.record2task.generator.e2b_environment import (
    PROFILE_PATH,
    E2BTaskEnvironment,
    ImageConfigError,
    ImageConfigs,
    RegistryImageConfigs,
    e2b_dockerfile,
    image_location,
)
from reef.record2task.generator.harbor import UnrunnableTask

from .test_generator_harbor import fake_harbor, generated

pytestmark = pytest.mark.unit

GOLANG_ENV = (("PATH", "/usr/local/go/bin:/usr/bin:/bin"), ("GOLANG_VERSION", "1.22.5"))
PYTHON_ENV = (("PATH", "/usr/local/bin:/usr/bin:/bin"), ("LANG", "C.UTF-8"))


class StandInImages(ImageConfigs):
    def __init__(self, error: Exception | None = None) -> None:
        self.environments = {"golang:1.22": GOLANG_ENV, "python:3.12-slim": PYTHON_ENV}
        self.error = error
        self.asked: list[str] = []

    def environment(self, image: str) -> tuple[tuple[str, str], ...]:
        self.asked.append(image)
        if self.error is not None:
            raise self.error
        if image not in self.environments:
            raise UnrunnableTask(f"the base image {image} cannot be pulled anonymously")
        return self.environments[image]


class StandInRegistry:
    """One image, golang:1.22: a bearer challenge, an index, its amd64 manifest, a config behind a redirect."""

    def __init__(self, failures: dict[str, list[int]] | None = None) -> None:
        self.failures = failures or {}
        self.requests: list[str] = []
        registry = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                registry.requests.append(self.path)
                path = urllib.parse.urlsplit(self.path).path
                is_authorized = self.headers.get("Authorization") == "Bearer t-1"
                if registry.failures.get(path):
                    self.answer(registry.failures[path].pop(0), {"errors": []})
                elif path == "/v2/":
                    challenge = f'Bearer realm="{registry.url}/token",service="stand-in"'
                    self.answer(401, {"errors": []}, {"WWW-Authenticate": challenge})
                elif path == "/token":
                    self.answer(200, {"token": "t-1"})
                elif not path.startswith("/storage/") and not is_authorized:
                    self.answer(401, {"errors": [{"code": "UNAUTHORIZED"}]})
                elif path == "/v2/library/golang/manifests/1.22":
                    platforms = (("arm64", "sha256:arm"), ("amd64", "sha256:amd"))
                    manifests = [{"digest": d, "platform": {"os": "linux", "architecture": a}} for a, d in platforms]
                    self.answer(200, {"manifests": manifests})
                elif path == "/v2/library/golang/manifests/sha256:amd":
                    self.answer(200, {"config": {"digest": "sha256:config"}})
                elif path == "/v2/library/golang/blobs/sha256:config":
                    self.answer(307, {}, {"Location": "/storage/config"})
                elif path == "/storage/config" and "Authorization" not in self.headers:
                    self.answer(200, {"config": {"Env": [f"{name}={value}" for name, value in GOLANG_ENV]}})
                else:
                    self.answer(404, {"errors": [{"code": "MANIFEST_UNKNOWN"}]})

            def answer(self, status: int, document: object, headers: dict[str, str] | None = None) -> None:
                body = json.dumps(document).encode()
                self.send_response(status)
                for name, value in {"Content-Length": str(len(body)), **(headers or {})}.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def reader(self) -> RegistryImageConfigs:
        return RegistryImageConfigs(registry_urls={"docker.io": self.url}, retry_s=0.0)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def registry():
    service = StandInRegistry()
    yield service
    service.close()


# ------------------------------------------------------------------------------------------ the registry


@pytest.mark.parametrize(
    ("image", "location"),
    [
        ("golang:1.22", ("docker.io", "library/golang", "1.22")),
        ("docker.io/owner/app", ("docker.io", "owner/app", "latest")),
        ("ghcr.io/owner/app:v1", ("ghcr.io", "owner/app", "v1")),
        ("localhost:5000/app@sha256:abc", ("localhost:5000", "app", "sha256:abc")),
    ],
)
def test_an_image_reference_names_its_registry_repository_and_tag_as_docker_resolves_it(image, location) -> None:
    assert image_location(image) == location


def test_the_config_is_read_through_the_challenge_the_amd64_manifest_and_the_redirect_once(registry) -> None:
    reader = registry.reader()
    assert reader.environment("golang:1.22") == GOLANG_ENV
    assert registry.requests == [
        "/v2/",
        "/token?scope=repository%3Alibrary%2Fgolang%3Apull&service=stand-in",
        "/v2/library/golang/manifests/1.22",
        "/v2/library/golang/manifests/sha256:amd",
        "/v2/library/golang/blobs/sha256:config",
        "/storage/config",
    ], "the token stays off the redirected storage address"
    assert reader.environment("golang:1.22") == GOLANG_ENV and len(registry.requests) == 6, "kept per reference"


def test_a_rate_limit_is_waited_out_and_a_registry_that_keeps_refusing_raises() -> None:
    registry = StandInRegistry({"/v2/library/golang/manifests/1.22": [429, 503]})
    try:
        assert registry.reader().environment("golang:1.22") == GOLANG_ENV
        registry.failures["/v2/library/golang/manifests/1.22"] = [429] * 4
        with pytest.raises(ImageConfigError, match=r"in 4 attempts: .*manifests/1.22 answered 429"):
            registry.reader().environment("golang:1.22")
    finally:
        registry.close()


def test_an_image_the_registry_does_not_serve_anonymously_is_unrunnable(registry) -> None:
    with pytest.raises(UnrunnableTask, match=r"golang:1.99 cannot be pulled anonymously .* answered 404"):
        registry.reader().environment("golang:1.99")


# -------------------------------------------------------------------------------------- the Dockerfile


def test_each_run_step_exports_the_image_env_and_the_dockerfile_env_and_args_so_far() -> None:
    dockerfile, environment = e2b_dockerfile(
        "FROM golang:1.22\n"
        "ARG MODE=fast\n"
        "RUN go version\n"
        "# the toolchain's own bin first\n"
        "ENV GOPATH=/go \\\n    PATH=/go/bin:$PATH NOTE='two words'\n"
        "ENV MODE=slow\n"
        'RUN ["go", "build"]\n'
        "RUN go test\n",
        StandInImages(),
    )
    image_path = "PATH=/usr/local/go/bin:/usr/bin:/bin GOLANG_VERSION=1.22.5"
    assert dockerfile.splitlines()[:6] == [
        "FROM golang:1.22",
        "ARG MODE=fast",
        f"RUN export MODE=fast {image_path} && go version",
        "ENV GOPATH=/go PATH=/go/bin:$PATH NOTE='two words'",
        "ENV MODE=slow",
        'RUN ["go", "build"]',
    ]
    assert dockerfile.splitlines()[6] == (
        "RUN export MODE=slow PATH=/go/bin:/usr/local/go/bin:/usr/bin:/bin GOLANG_VERSION=1.22.5 GOPATH=/go "
        "NOTE='two words' && go test"
    ), "a later ENV wins over the ARG of its name, and $PATH is the image's"
    assert environment == {
        "PATH": "/go/bin:/usr/local/go/bin:/usr/bin:/bin",
        "GOLANG_VERSION": "1.22.5",
        "GOPATH": "/go",
        "NOTE": "two words",
        "MODE": "slow",
    }, "the container gets the ENV, not the ARG"


def test_from_and_workdir_get_their_variables_expanded() -> None:
    images = StandInImages()
    dockerfile, _ = e2b_dockerfile(
        "ARG TAG=1.22\nFROM --platform=linux/amd64 golang:${TAG} AS build\nARG TAG\nARG APP=/srv/app-$TAG\n"
        "WORKDIR ${APP}/src\n",
        images,
    )
    assert images.asked == ["golang:1.22"]
    assert dockerfile.splitlines()[1] == "FROM golang:1.22" and "WORKDIR /srv/app-1.22/src" in dockerfile


@pytest.mark.parametrize(
    ("user", "tail"), [("app", ["USER app", "USER root", "STEP", "USER app"]), ("root", ["USER root", "STEP"])]
)
def test_the_profile_step_writes_the_container_env_as_root(user: str, tail: list[str]) -> None:
    dockerfile, _ = e2b_dockerfile(f"FROM golang:1.22\nENV CGO_ENABLED=0\nUSER {user}\n", StandInImages())
    lines = dockerfile.splitlines()
    step = next(line for line in lines if PROFILE_PATH in line)
    assert lines[-len(tail) :] == [step if line == "STEP" else line for line in tail]
    assert step.endswith(f"echo '. {PROFILE_PATH}' >> /etc/bash.bashrc && echo '. {PROFILE_PATH}' >> /root/.bashrc")
    profile = base64.b64decode(step.split("echo ", 1)[1].split(" |", 1)[0]).decode()
    assert profile == "export PATH=/usr/local/go/bin:/usr/bin:/bin GOLANG_VERSION=1.22.5 CGO_ENABLED=0\n"


def test_a_multi_stage_dockerfile_is_unrunnable_before_any_registry_call() -> None:
    images = StandInImages()
    with pytest.raises(UnrunnableTask, match="single-stage Dockerfile; this one has 2 FROM lines"):
        e2b_dockerfile("FROM golang:1.22 AS build\nRUN go build\nFROM python:3.12-slim\nRUN true\n", images)
    assert images.asked == []


# ------------------------------------------------------------------------------------- the prepared copy


def test_the_copy_states_the_environment_and_the_task_stays_as_written(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    written = read_harbor_task(root)
    environment = E2BTaskEnvironment(tmp_path / "e2b", StandInImages())
    copy = environment.prepare(root)
    assert copy == tmp_path / "e2b" / root.name
    header = "export PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8"
    for script in ("tests/test.sh", "solution/solve.sh"):
        assert (copy / script).read_text().splitlines()[:2] == ["#!/bin/sh", header]
    assert (copy / "environment" / "Dockerfile").read_text().startswith(f"FROM python:3.12-slim\nRUN {header} && ")
    assert read_harbor_task(root) == written, "the written task keeps its files and its digest"
    first = {path.relative_to(copy): path.read_bytes() for path in copy.rglob("*") if path.is_file()}
    environment.prepare(root)
    assert {path.relative_to(copy): path.read_bytes() for path in copy.rglob("*") if path.is_file()} == first


def test_a_task_on_a_prebuilt_image_is_unrunnable(tmp_path: Path) -> None:
    task = harbor_task(generated())
    config = {**task.config, "environment": {**task.config["environment"], "docker_image": "python:3.12-slim"}}
    root = write_harbor_task(dataclasses.replace(task, config=config), tmp_path / "tasks")
    with pytest.raises(UnrunnableTask, match=r"prebuilt environment\.docker_image"):
        E2BTaskEnvironment(tmp_path / "e2b", StandInImages()).prepare(root)


# ------------------------------------------------------------------------------------ checks and plays


def test_the_checks_run_the_copy_in_e2b(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0})
    result = oracle_check(root, harbor=harbor, environment=E2BTaskEnvironment(tmp_path / "e2b", StandInImages()))
    assert result.is_solvable
    for agent in ("oracle", "nop"):
        arguments = json.loads((root.parent / ".harbor-jobs" / root.name / agent / "arguments.json").read_text())
        assert arguments[arguments.index("-e") + 1] == "e2b"
        assert arguments[arguments.index("-p") + 1] == str(tmp_path / "e2b" / root.name)


def test_an_unrunnable_task_is_refused_and_an_unread_registry_leaves_the_check_undone(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    harbor = fake_harbor(tmp_path, {"oracle": 1.0, "nop": 0.0})
    missing = StandInImages()
    missing.environments = {}
    result = oracle_check(root, harbor=harbor, environment=E2BTaskEnvironment(tmp_path / "e2b", missing))
    assert not result.is_solvable and result.reason.startswith("the base image python:3.12-slim cannot be pulled")
    assert not (root.parent / ".harbor-jobs" / root.name / "oracle").exists(), "harbor never ran"
    unread = StandInImages(error=ImageConfigError("could not read the config of python:3.12-slim in 4 attempts"))
    with pytest.raises(OracleUnavailable, match="could not read the config"):
        oracle_check(root, harbor=harbor, environment=E2BTaskEnvironment(tmp_path / "e2b", unread))


class StandInLab(TaskLab):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> EpisodeRow:
        self.calls.append({"task_path": task_path, "overrides": dict(overrides)})
        return EpisodeRow(rewards={"reward": 1.0})


def test_the_plays_run_the_copy_in_e2b_and_name_the_task(tmp_path: Path) -> None:
    root = write_harbor_task(harbor_task(generated()), tmp_path / "tasks")
    lab = StandInLab()
    plays = ReefTaskPlays(
        reef_url="http://127.0.0.1:9",
        work_dir=tmp_path / "play",
        environment=E2BTaskEnvironment(tmp_path / "e2b", StandInImages()),
        lab=lab,
    )
    played = plays.play(
        root,
        scenario="s",
        model="m",
        arm="plain",
        plays=2,
        is_reporting=False,
        extra_instruction_paths=(),
        tags={},
    )
    assert [call["task_path"] for call in lab.calls] == [tmp_path / "e2b" / root.name] * 2
    assert all(call["overrides"] == {"environment": {"type": "e2b"}} for call in lab.calls)
    assert [(play.task_path, play.name, play.reward) for play in played] == [(root, root.name, 1.0)] * 2
