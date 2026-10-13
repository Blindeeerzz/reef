"""Harbor tasks prepared for Harbor's E2B environment, with their base image's environment read from its registry.

Harbor (0.20 through 0.23) hands E2B a task's Dockerfile as written, and E2B's own builder departs from Docker
wherever a task depends on its environment variables:

- RUN steps run without the base image's ENV (golang's ``PATH`` entry ``/usr/local/go/bin``) and without the
  Dockerfile's own ENV and ARG;
- WORKDIR is taken literally, ``${VAR}`` and all;
- the verifier, the reference solution and the agent's shell run without any of that environment.

``E2BTaskEnvironment`` therefore hands Harbor a copy of the task that states the environment Docker would apply.
The checks and the plays run the same copy, so E2B builds one template per task; the written task stays as it
is, since its digest names it in reports.
"""

from __future__ import annotations

import base64
import email.message
import json
import re
import shlex
import shutil
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.record2task.generator.harbor import TaskEnvironment, UnrunnableTask, dockerfile_logical_lines

DOCKER_HUB = "docker.io"
REGISTRY_URLS = {DOCKER_HUB: "https://registry-1.docker.io"}
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
MANIFEST_TYPES = ("application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json")
REFUSED_STATUSES = (401, 403, 404)
REGISTRY_ATTEMPTS = 4
REGISTRY_RETRY_S = 5.0
MAX_RETRY_AFTER_S = 60.0
REGISTRY_TIMEOUT_S = 30.0
PROFILE_PATH = "/etc/profile.d/reef-image-env.sh"
VARIABLE_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_]\w*)(?::(?P<modifier>[-+])(?P<word>[^}]*))?\}|(?P<bare>[A-Za-z_]\w*))"
)
SHELL_NAME_PATTERN = re.compile(r"[A-Za-z_]\w*")
CHALLENGE_FIELD_PATTERN = re.compile(r'(\w+)="([^"]*)"')


class ImageConfigError(RuntimeError):
    """A base image's config could not be read from its registry, so the environment it declares is unknown."""


class ImageConfigs(ABC):
    """The environment variables a base image's config declares, which Docker starts its containers with."""

    @abstractmethod
    def environment(self, image: str) -> tuple[tuple[str, str], ...]:
        """``(name, value)`` pairs in the config's order; ``UnrunnableTask`` when E2B cannot pull ``image``."""


def image_location(image: str) -> tuple[str, str, str]:
    """The registry, repository and tag or digest an image reference names, resolved the way Docker resolves it."""
    name, _, digest = image.partition("@")
    first, slash, rest = name.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = DOCKER_HUB, name
    tag = "latest"
    if ":" in repository.rsplit("/", 1)[-1]:
        repository, tag = repository.rsplit(":", 1)
    if registry == DOCKER_HUB and "/" not in repository:
        repository = f"library/{repository}"
    return registry, repository, digest or tag


class RegistryImageConfigs(ImageConfigs):
    """Image configs read from their registries over HTTPS, anonymously and without pulling; kept per reference.

    A 429, a server error or a dropped connection is tried again after the registry's ``Retry-After`` or a
    doubling wait, and then raises ``ImageConfigError``. Harbor gives E2B no registry credentials and E2B runs
    linux/amd64, so an image the registry refuses anonymously (401, 403, 404) or has no linux/amd64 build of is
    ``UnrunnableTask``. ``registry_urls`` maps a registry name (``docker.io``) to its API address.
    """

    def __init__(self, *, registry_urls: Mapping[str, str] = REGISTRY_URLS, retry_s: float = REGISTRY_RETRY_S) -> None:
        self.registry_urls = registry_urls
        self.retry_s = retry_s
        self.cache: dict[str, tuple[tuple[str, str], ...]] = {}

    def environment(self, image: str) -> tuple[tuple[str, str], ...]:
        if image not in self.cache:
            self.cache[image] = self.read_environment(image)
        return self.cache[image]

    def read_environment(self, image: str) -> tuple[tuple[str, str], ...]:
        """The image's manifest (the linux/amd64 one of an index), then the config blob it names, then its Env."""
        registry, repository, reference = image_location(image)
        root = self.registry_urls.get(registry, f"https://{registry}")
        token = self.pull_token(root, repository, image)
        api = f"{root}/v2/{repository}"
        manifest = self.registry_document(
            f"{api}/manifests/{reference}", accept=", ".join((*INDEX_TYPES, *MANIFEST_TYPES)), token=token, image=image
        )
        if "manifests" in manifest:
            # An index entry's platform is optional in the OCI format.
            digests = [
                entry["digest"]
                for entry in manifest["manifests"]
                if {"os": "linux", "architecture": "amd64"}.items() <= entry.get("platform", {}).items()
            ]
            if not digests:
                raise UnrunnableTask(f"the base image {image} has no linux/amd64 build, the platform E2B runs")
            manifest = self.registry_document(
                f"{api}/manifests/{digests[0]}", accept=", ".join(MANIFEST_TYPES), token=token, image=image
            )
        document = self.registry_document(
            f"{api}/blobs/{manifest['config']['digest']}", accept="application/json", token=token, image=image
        )
        # An image config may omit its config and Env, and Docker writes an empty Env as null.
        variables = document.get("config", {}).get("Env") or []
        return tuple((name, value) for name, _, value in (variable.partition("=") for variable in variables))

    def pull_token(self, root: str, repository: str, image: str) -> str | None:
        """An anonymous pull token for ``repository`` when the registry challenges for one; None when it needs none."""
        status, headers, _ = self.fetch(urllib.request.Request(f"{root}/v2/"), image)
        if 200 <= status < 300:
            return None
        challenge = headers.get("WWW-Authenticate", "")
        if status != 401 or not challenge.lower().startswith("bearer "):
            raise ImageConfigError(f"reading {image}: {root}/v2/ answered {status} without a bearer challenge")
        fields = dict(CHALLENGE_FIELD_PATTERN.findall(challenge))
        query = {"scope": f"repository:{repository}:pull"}
        if "service" in fields:
            query["service"] = fields["service"]
        document = self.registry_document(
            f"{fields['realm']}?{urllib.parse.urlencode(query)}", accept="application/json", token=None, image=image
        )
        # The token specification allows either name.
        return document.get("token") or document["access_token"]

    def registry_document(self, url: str, *, accept: str, token: str | None, image: str) -> dict[str, Any]:
        """One JSON document from the registry; a refusal of anonymous access means E2B cannot pull the image."""
        request = urllib.request.Request(url, headers={"Accept": accept})
        if token is not None:
            # Not carried to a redirect: a blob's storage address is signed and refuses a second credential.
            request.add_unredirected_header("Authorization", f"Bearer {token}")
        status, _, body = self.fetch(request, image)
        if status in REFUSED_STATUSES:
            raise UnrunnableTask(
                f"the base image {image} cannot be pulled anonymously ({url} answered {status}), so E2B cannot build it"
            )
        if status != 200:
            raise ImageConfigError(f"reading {image}: {url} answered {status}")
        return json.loads(body)

    def fetch(self, request: urllib.request.Request, image: str) -> tuple[int, email.message.Message, bytes]:
        """The status, headers and body of a GET; a 429, a server error or a dropped connection is tried again."""
        failure = ""
        for attempt in range(REGISTRY_ATTEMPTS):
            delay = self.retry_s * 2**attempt
            try:
                with urllib.request.urlopen(request, timeout=REGISTRY_TIMEOUT_S) as response:
                    return response.status, response.headers, response.read()
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    return exc.code, exc.headers, exc.read()
                failure = f"{request.full_url} answered {exc.code}"
                retry_after = exc.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = min(float(retry_after), MAX_RETRY_AFTER_S)
            except OSError as exc:
                failure = f"{request.full_url}: {exc}"
            if attempt + 1 < REGISTRY_ATTEMPTS:
                time.sleep(delay)
        raise ImageConfigError(f"could not read the config of {image} in {REGISTRY_ATTEMPTS} attempts: {failure}")


def expand_variables(text: str, variables: Mapping[str, str]) -> str:
    """``$NAME``, ``${NAME}``, ``${NAME:-word}`` and ``${NAME:+word}`` expanded as Docker does; unset is empty."""

    def value_of(match: re.Match[str]) -> str:
        value = variables.get(match["braced"] or match["bare"], "")
        if match["modifier"] == "-":
            return value or expand_variables(match["word"], variables)
        if match["modifier"] == "+":
            return expand_variables(match["word"], variables) if value else ""
        return value

    return VARIABLE_PATTERN.sub(value_of, text)


def instruction_words(arguments: str) -> list[str]:
    """The words of an ENV or ARG instruction's arguments, quotes removed."""
    try:
        return shlex.split(arguments)
    except ValueError as exc:
        raise UnrunnableTask(f"Dockerfile: {arguments[:90]}: {exc}") from exc


def export_line(variables: Mapping[str, str]) -> str:
    """One ``export`` of the variables with shell-quoted values; empty when there are none.

    A name the shell cannot export (Docker accepts any) is left out rather than failing the whole line.
    """
    words = [f"{name}={shlex.quote(value)}" for name, value in variables.items() if SHELL_NAME_PATTERN.fullmatch(name)]
    return f"export {' '.join(words)}" if words else ""


def e2b_dockerfile(text: str, images: ImageConfigs) -> tuple[str, dict[str, str]]:
    """The Dockerfile that E2B builds with Docker's environment, and the environment Docker starts the container with.

    The Dockerfile is rewritten from its logical lines (continuations joined, comments dropped), so a multi-line
    ENV reads as one. Variables follow Docker: an ARG before FROM serves FROM and an ARG that names it again; an
    ARG after FROM and every ENV serve the lines that follow; an ENV wins over an ARG of the same name; the
    container gets the image's ENV and the Dockerfile's ENV, not its ARG.
    """
    lines = dockerfile_logical_lines(text)
    stages = sum(1 for line in lines if line.split(None, 1)[0].upper() == "FROM")
    if stages != 1:
        raise UnrunnableTask(f"E2B builds a single-stage Dockerfile; this one has {stages} FROM lines")
    global_args: dict[str, str] = {}
    build_args: dict[str, str] = {}
    environment: dict[str, str] = {}
    image: str | None = None
    user = "root"
    prepared: list[str] = []
    for line in lines:
        parts = line.split(None, 1)
        keyword, rest = parts[0].upper(), parts[1] if len(parts) > 1 else ""
        variables = global_args if image is None else {**build_args, **environment}
        if keyword == "ARG":
            scope = global_args if image is None else build_args
            for word in instruction_words(rest):
                name, has_default, default = word.partition("=")
                if has_default:
                    scope[name] = expand_variables(default, variables)
                elif image is not None and name in global_args:
                    scope[name] = global_args[name]
        elif keyword == "FROM":
            names = [word for word in rest.split() if not word.startswith("--")]
            # E2B pulls the FROM line's value as it stands, flags and variables included: it gets the image alone.
            image = expand_variables(names[0], global_args)
            environment = dict(images.environment(image))
            line = f"FROM {image}"
        elif keyword == "ENV":
            words = instruction_words(rest)
            pairs = (
                [(words[0], " ".join(words[1:]))]
                if words and "=" not in words[0]
                else [(name, value) for name, _, value in (word.partition("=") for word in words)]
            )
            environment.update({name: expand_variables(value, variables) for name, value in pairs})
        elif keyword == "WORKDIR":
            line = f"WORKDIR {expand_variables(rest, variables)}"
        elif keyword == "USER":
            user = rest
        elif keyword == "RUN" and not rest.startswith("["):
            exports = export_line(variables)
            line = f"RUN {exports} && {rest}" if exports else line
        prepared.append(line)
    # The agent's shell (terminus-2 starts bash in tmux) reads these; the RUN step needs root to write them.
    profile = base64.b64encode(f"{export_line(environment)}\n".encode()).decode()
    step = (
        f"RUN mkdir -p /etc/profile.d && echo {profile} | base64 -d > {PROFILE_PATH} && "
        f"echo '. {PROFILE_PATH}' >> /etc/bash.bashrc && echo '. {PROFILE_PATH}' >> /root/.bashrc"
    )
    is_root = user.split(":", 1)[0] in ("root", "0")
    prepared.extend([step] if is_root else ["USER root", step, f"USER {user}"])
    return "\n".join(prepared) + "\n", environment


class E2BTaskEnvironment(TaskEnvironment):
    """Harbor's E2B environment, handed a copy of each task under ``root`` that states the image's environment."""

    name = "e2b"

    def __init__(self, root: Path, images: ImageConfigs | None = None) -> None:
        self.root = root
        self.images = images if images is not None else RegistryImageConfigs()

    def prepare(self, task_path: Path) -> Path:
        """``root/<task name>``, written afresh from the task each time, so one task always gives the same copy."""
        config = tomllib.loads((task_path / "task.toml").read_text(encoding="utf-8"))
        if config.get("environment", {}).get("docker_image"):
            raise UnrunnableTask(
                "E2B starts a prebuilt environment.docker_image without the image's environment, and the "
                "generator cannot give it to the agent's shell"
            )
        dockerfile, environment = e2b_dockerfile(
            (task_path / "environment" / "Dockerfile").read_text(encoding="utf-8"), self.images
        )
        header = export_line(environment)
        copy = self.root / task_path.name
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(task_path, copy)
        (copy / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        # A task need not ship a reference solution.
        for script in (copy / "tests" / "test.sh", copy / "solution" / "solve.sh"):
            if not script.is_file():
                continue
            # After the shebang, so the interpreter line stays first.
            lines = script.read_text(encoding="utf-8").splitlines()
            lines.insert(1 if lines and lines[0].startswith("#!") else 0, header)
            script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return copy
