Connector HTTP protocol
=======================

This contract is for a receiver that connects a console to Reef through
``reef connect``. It carries the native HTTP API without business-command
translation. For operator setup and policy examples, see the
`CLI reference <cli.rst#connect-to-the-api-platform>`__.

Connection and authorization
----------------------------

The connector pairs through ``POST /api/connector/pair``. Its JSON body has
``protocol: 2``, ``instance_id``, and ``name``. The receiver returns
``verification_uri``, ``user_code``, ``device_token``, and ``expires_in``.
The expiry must be positive and at most 600 seconds. The verification URL
must share the receiver's origin. The connector polls
``GET /api/connector/pair`` with the device token as a Bearer credential.
Approval returns ``status: "approved"`` and ``runtime_id``. The device token
then becomes the connection credential. Pending responses use a different
status. The connector saves the credential only after approval.

The connector opens an authenticated outbound WebSocket at
``/api/connector/stream``. HTTPS endpoints use WSS; loopback HTTP endpoints
use WS. Redirects are rejected. Handshake HTTP 401 or 403 ends the connector
and requires reauthorization. Other handshake 4xx responses are fatal.
Network failures reconnect with delays from one to thirty seconds.

Messages are JSON objects in text WebSocket frames, at most 2097152 bytes.
The connector first sends ``hello``:

.. code:: json

   {
     "type": "hello",
     "protocol": 2,
     "instance_id": "a121dc8d-fb21-471c-8afb-09df5611743a",
     "runtime_version": "0.1.0",
     "capabilities": {"record_bodies": false, "inference": false},
     "limits": {
       "max_concurrent_requests": 4,
       "max_request_bytes": 1048576,
       "max_message_bytes": 2097152,
       "response_chunk_bytes": 32768,
       "request_timeout_seconds": 600
     }
   }

``runtime_version`` is the installed distribution version. Limits reflect
the local policy. Within fifteen seconds, the receiver must send
``{"type": "ready", "protocol": 2, "limits": {...}}`` with exactly the
advertised integer limits. Unsupported protocols or limits end the connector
before dispatch. These capability flags do not replace route authorization.

The connector sends ``{"type": "heartbeat", "protocol": 2}`` every fifteen
seconds, subject to connection backpressure. WebSocket ping/pong checks run
separately. Heartbeats contain no scenario summaries.

Requests
--------

After ``ready``, the receiver can send this request:

.. code:: json

   {
     "type": "request",
     "id": "9e12e095-2c43-49ab-a257-8c49110b28cb",
     "method": "GET",
     "path": "/reef/scenarios/demo/releases",
     "query": "limit=20",
     "headers": {"accept": "application/json"},
     "body": ""
   }

``id`` must be a UUID. Supported methods are ``GET``, ``POST``, and ``DELETE``.
Only POST can carry a body. ``body`` is base64; its decoded size must fit
``max_request_bytes``. ``query``, ``headers``, and ``body`` default to empty.
Unknown request fields are rejected.

``path`` is a relative ``/reef/`` or ``/v1/`` path, at most 2048 characters.
``query`` is at most 8192 characters without a leading question mark.
Use URL-encoded characters. Alternate addresses, dot segments, encoded path
separators, nested escapes, fragments, and control characters are rejected.
Scenario selectors in the path, ``x-reef-scenario`` header, query, and JSON
body must agree. Creating a scenario also checks the JSON ``name`` field.
The resolved scenario is supplied to Reef through ``x-reef-scenario``.

Request header names are case-insensitive. Permitted headers are ``accept``,
``content-type``, ``anthropic-version``, ``anthropic-beta``, ``x-request-id``,
``x-reef-scenario``, ``x-reef-release-id``, ``traceparent``, and ``tracestate``.
Other headers are removed, including remote authorization and cookies.
Headers named by ``Connection`` are also removed. The connector supplies
the local service credential and requests identity response encoding.

Responses and cancellation
--------------------------

Each response starts with ``response`` containing ``id``, native HTTP
``status``, and ``headers`` as a list of name/value pairs. It continues with
zero or more ``chunk`` messages containing ``id`` and base64 ``body``.
Decoded chunks are at most 32768 bytes. ``end`` contains ``id`` and marks
completion. Messages for concurrent requests can interleave; each request's
chunks remain ordered. Native HTTP error bodies follow the same sequence.
Concatenating decoded chunks reconstructs the native response bytes,
including SSE delimiters. Responses cannot use compressed encoding.

Permitted response headers are ``content-type``, ``cache-control``,
``content-disposition``, ``retry-after``, ``x-request-id``,
``x-reef-agent-record-id``, ``x-reef-release-id``, and
``x-reef-artifact-version``. Cookies and connection-specific headers are
removed. External redirects produce a protocol error and are never followed.

``{"type": "cancel", "id": "..."}`` cancels local forwarding. An accepted
Reef operation can continue after cancellation. A deadline includes local
response reading and remote sending. Every send has a fifteen-second limit.
The connector reads another response chunk only after its previous send
completes, so a slow receiver applies backpressure.

Protocol failures use ``error`` with ``id``, ``code``, and ``outcome``.
An invalid or missing ID can produce ``id: null``. An error can follow partial
response chunks; it replaces ``end``. Treat any response without ``end``
as incomplete.

.. list-table::
   :header-rows: 1

   * - Code
     - Meaning
   * - ``request_rejected``
     - Invalid request or local authorization denied; not dispatched.
   * - ``capacity_exceeded``
     - All permitted request slots are occupied; not dispatched.
   * - ``already_active``
     - This read request ID is still active; no second dispatch.
   * - ``already_dispatched``
     - A mutation ID already has a saved dispatch record.
   * - ``id_conflict``
     - A mutation ID was reused with different request content.
   * - ``cancelled``, ``deadline_exceeded``, ``connection_interrupted``
     - Forwarding stopped after dispatch; outcome unknown.
   * - ``external_redirect``, ``unsupported_encoding``
     - Reef returned an unsupported response; outcome unknown.
   * - ``not_active``
     - No forwarding task matches the cancelled ID; outcome unknown.

``outcome`` is ``not_dispatched``, ``unknown``, or ``completed``. For duplicate
mutation IDs, it reports the saved outcome. ``completed`` means the full HTTP
response reached ``end``; it does not mean the requested operation succeeded.
Mutation IDs and request hashes persist across restarts. The connector never
replays them or saves response bodies. After an uncertain outcome, the client
must check native receipts or state before submitting another operation.
