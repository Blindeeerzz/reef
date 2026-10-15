Reef CLI: serve and connect
==========================

``reef serve`` reads a deployment config and starts every
process the config declares, in dependency order. ``reef connect`` optionally
links an existing runtime to your API platform account. (The wheel also installs
``reef-native`` and ``reef-terminus``, the loop runners those two harness
adapters launch per episode; nothing calls them by hand.)

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml

``reef serve`` runs in the foreground and holds the terminal until Ctrl-C. Each
service's ``ready`` probe must pass before the next one starts; when they are
all up, Reef blocks, and a watchdog tears the stack down if any process exits
unexpectedly.

Once the stack is running, Ctrl-C requests graceful shutdown. Pressing it
again during shutdown skips the remaining 30-second grace period and proceeds
to forced cleanup of the managed service processes.

.. config::

   -c, --config | optional config file. No file is loaded unless explicitly selected.
   --recipe NAME | start a built in recipe's profile instead of a config file. Today: ``reefine``.
   --model [PROVIDER/]MODEL | the upstream model. An ``ollama/`` or ``openai/`` prefix fills the endpoint and the key; any other spelling is the model ID as is.
   --print-config | print every resolved setting with its source (file, command line, environment, automatic, default) and exit without downloading models or starting services. Credentials are masked.
   --help | the command list
   -V, --version | the installed reef version. Takes no command: ``reef --version``.

Starting a recipe's profile
---------------------------

A built in recipe can carry a profile: one deployment config that is the same
for every deployment of that recipe except the model. ``--recipe`` starts it
without a file of your own:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b

Select a file with ``-c`` or a built-in profile with ``--recipe``. With neither,
Reef uses CLI inference settings and does not discover ``$REEF_CONFIG`` or
``reef.yaml``. The profile selector is a launcher option; configuration values
use their full public namespaces as shown above.

The optional ``--model`` shorthand remains supported: ``ollama/`` fills
``http://127.0.0.1:11434`` and a placeholder key; ``openai/`` fills
``https://api.openai.com`` and reads ``REEF_UPSTREAM_API_KEY``. Other prefixes
and bare values are model IDs and need an upstream URL.
An explicit ``--inference.upstream-url`` or ``--inference.upstream-api-key`` override wins over
the prefix. The ``reefine`` profile ships its proposer and evaluator in the
wheel, so it runs from an installed package; it listens on
``127.0.0.1:8901`` without authentication unless ``REEF_TOKEN`` is set, and keeps its state under
``.reef/reefine/``. To change anything else, copy
``reef/service/profiles/reefine.yaml`` and pass the copy with ``-c``.

Overriding config values
------------------------

Canonical CLI paths match the sections of a configuration file. Declared fields use the
same type parser for both inputs; explicit CLI values override YAML. Opaque
component objects support leaf overrides, validated by their owning component.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct \
     --training.config.checkpoint_dir /tmp/ckpt

Use public namespaces to move a stack's state without editing its config.
Bare and ``reef.*`` aliases remain accepted:

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --storage.agent-record-dir .reef/agent-record \
     --storage.artifact-work-dir .reef/artifact-work \
     --storage.artifact-cache-dir .reef/artifact-cache

Where it writes
---------------

Each service gets a log and a PID file under ``reef.run-dir``, which defaults
to ``.reef/run``. Reef writes its own state, including records, commit logs,
and the Git-backed release chain, to the ``storage.*`` paths in the config.

Connect to the API platform
--------------------------

``reef connect`` forwards authorized HTTP requests to your Reef service.
Direct connections and connector connections use the same API fields and
response bodies. The receiver must implement the
`connector protocol <connector-protocol.rst>`__.

First, create a local ``policy.json``. This example permits two operations
on the ``demo`` scenario:

.. code:: json

   {
     "scenarios": ["demo"],
     "routes": [
       {"method": "GET", "path": "/reef/scenarios/{scenario}/releases"},
       {"method": "POST", "path": "/reef/scenarios/{scenario}/update"}
     ]
   }

Start Reef separately, then connect it:

.. code:: bash

   uv run reef connect \
     --url http://127.0.0.1:9000 \
     --platform http://localhost:3000 \
     --http-policy policy.json \
     --name workstation \
     --no-browser --foreground

Replace both addresses with your service and receiver addresses. Open the
printed sign-in link and enter the device code. ``--no-browser`` prints the
link without opening it. ``--foreground`` keeps the connector in this terminal;
omit it to run in the background. The connection is outbound, so it needs no
inbound port or browser CORS configuration.

If omitted, ``--url`` defaults to ``http://127.0.0.1:8900`` and ``--platform``
defaults to ``https://api.reefinfra.ai``. URLs require HTTPS except for loopback
HTTP. Set ``REEF_TOKEN`` if the local service requires authentication. Use
``--reef-token-env VARIABLE`` to select another environment variable.
The local service token stays on this machine. Do not put tokens in URLs
or command-line arguments.

Local permissions
~~~~~~~~~~~~~~~~~

The connector saves a copy of the policy for subsequent starts. Pass
``--http-policy`` again to replace it while the connector is stopped.
Pairing requires this policy. A browser approval alone grants no route access.

* ``scenarios`` lists allowed scenario names. Use ``["*"]`` to allow all.
  Global discovery through ``GET /reef/scenarios`` and ``GET /reef/status``
  requires this all-scenarios grant and a matching route.
* ``routes`` lists allowed ``method`` and ``path`` pairs. Supported methods
  are ``GET``, ``POST``, and ``DELETE``. Paths match exactly, with optional
  ``{scenario}`` substitution or a final ``/*`` for descendant paths.
  A final ``/*`` does not match the parent path itself.
* ``allow_record_bodies`` defaults to ``false``. Set it to ``true`` for routes
  with a ``records``, ``requests``, or ``trajectories`` path segment.
* ``allow_inference`` defaults to ``false``. Set it to ``true`` for ``/v1/``
  routes and scenario evaluation inference routes. A matching route and
  scenario authorization are still required.

Authorized responses leave this machine in full, including native error
bodies. Select routes with that access in mind. The connector rejects
conflicting scenario selectors, path escapes, external redirects, and
unapproved routes. Provider configuration remains part of the native scenario
API; the connector does not translate it into separate commands.

Defaults are four concurrent requests, 1 MiB per request body, and a
600-second request deadline. Set ``max_concurrent_requests`` from 1 to 16,
``max_request_bytes`` from 1 to 1048576, and ``request_timeout_seconds``
from 1 to 3600 in the policy. Streamed responses arrive as ordered chunks.
Slow receivers apply backpressure to the local response reader.

Start Reef with the connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this command from your Reef project directory. Put ``--serve`` last:

.. code:: bash

   uv run reef connect \
     --url http://127.0.0.1:9000 \
     --platform http://localhost:3000 \
     --http-policy policy.json \
     --no-browser \
     --serve --recipe reefine --model ollama/qwen3

Reef starts after approval. ``--url`` must specify a loopback HTTP address
and port. It supplies the host and port to ``reef serve``; do not pass
``--reef.host`` or ``--reef.port`` after ``--serve``. The connector refuses
an address already in use. Reef receives the local service token as
``REEF_TOKEN`` and writes output to ``serve.log`` in the connection directory.
The connector stops this service when it exits. It does not restart a service
that exits on its own. Without ``--serve``, the connector only checks whether
Reef answers and prints any readiness warning before pairing.

Lifecycle and saved state
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same ``--url`` and ``--platform`` with ``--status`` or ``--stop``.
Rerun your connection command to restart. Stopping retains authorization and
the saved policy. It leaves a separately managed Reef service running.
After revocation, restart with ``--reauthorize`` to pair again.

Each endpoint pair has a private directory under ``~/.reef/connections/``.
``--state-dir PATH`` selects another directory. It stores the instance UUID,
policy, credentials, a process lock, logs, and ``requests.sqlite3``.
On POSIX systems, the directory is mode 0700; credentials, request state, and
logs are mode 0600. Preserve this directory to retain identity and mutation
history. Do not share it between running machines.

The request database stores mutation IDs, request hashes, and outcomes.
It stores no request or response bodies. Reusing a mutation ID never repeats
the operation. An interrupted mutation remains ``unknown`` across restarts.
Check native receipts or scenario state before submitting a new operation.
An HTTP error is a completed response; it retains its native status and body.
Cancellation cannot undo work already accepted by Reef.

The connector reconnects after network failures. It does not install an OS
startup service. Use ``--foreground`` with your process supervisor for restart
after a machine reboot.
