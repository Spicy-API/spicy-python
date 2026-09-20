from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import pathlib
import unittest
import urllib.parse

# Load straight from the source tree rather than depending on an installed package - the tests
# must run before `pip install -e .`.
MODULE_PATH = pathlib.Path(__file__).parents[1] / "src" / "spicyapi" / "_client.py"
SPEC = importlib.util.spec_from_file_location("spicyapi_client_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
client_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client_module)


def response(data: object, status: int = 200, headers: dict[str, str] | None = None):
    return status, headers or {}, json.dumps(data).encode()


class QueueTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, body, timeout))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ReferenceClientTests(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual(client_module.API_BASE_URL, "https://api.spicyapi.ai/api/v1")
        self.assertEqual(client_module.REQUEST_TIMEOUT_SECONDS, 30.0)
        self.assertEqual(client_module.WAIT_TIMEOUT_SECONDS, 600.0)
        self.assertEqual(
            client_module.TERMINAL_STATES,
            frozenset({"succeeded", "failed", "canceled", "expired"}),
        )

    def test_models_auth_query_and_encoded_id(self):
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": {"total": 0, "items": []}}),
            response({"code": 200, "msg": "success", "data": {"model": "provider/model"}}),
        ])
        client = client_module.SpicyClient("test-key", transport=transport, max_retries=0)
        client.list_models(modality="video", include_schema=True)
        client.get_model("provider/model")
        self.assertRegex(transport.calls[0][1], r"/models\?modality=video&includeSchema=1$")
        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer test-key")
        self.assertTrue(transport.calls[1][1].endswith("/models/provider%2Fmodel"))
        self.assertGreater(transport.calls[0][4], 29.9)
        self.assertLessEqual(transport.calls[0][4], 30.0)

    def test_create_keeps_idempotency_key(self):
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": {
                "taskId": "job-1", "state": "queued", "estimatedCost": "0.1"}}, 202),
        ])
        client = client_module.SpicyClient("test-key", transport=transport, max_retries=0)
        client.create_task(
            model="provider/model",
            input_data={"prompt": "hello"},
            idempotency_key="logical-request-1",
        )
        self.assertEqual(transport.calls[0][2]["Idempotency-Key"], "logical-request-1")
        self.assertEqual(len(transport.calls), 1)

    def test_retry_creates_new_task_with_idempotency_key(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "taskId": "job-retry",
                    "sourceTaskId": "job-source",
                    "state": "queued",
                    "estimatedCost": "0.1",
                },
            }),
        ])
        client = client_module.SpicyClient("test-key", transport=transport, max_retries=0)
        retried = client.retry_task("job-source", "retry-logical-1")
        self.assertEqual(retried["sourceTaskId"], "job-source")
        self.assertTrue(transport.calls[0][1].endswith("/jobs/retry"))
        self.assertEqual(transport.calls[0][2]["Idempotency-Key"], "retry-logical-1")
        self.assertEqual(json.loads(transport.calls[0][3]), {"taskId": "job-source"})

    def test_wait_returns_all_terminal_states(self):
        for terminal in client_module.TERMINAL_STATES:
            with self.subTest(terminal=terminal):
                clock = [0.0]
                delays = []
                transport = QueueTransport([
                    response({"code": 200, "msg": "success", "data": {
                        "taskId": "job-1", "model": "provider/model", "state": "queued"}}),
                    response({"code": 200, "msg": "success", "data": {
                        "taskId": "job-1", "model": "provider/model", "state": "running"}}),
                    response({"code": 200, "msg": "success", "data": {
                        "taskId": "job-1", "model": "provider/model", "state": terminal}}),
                ])

                # Default arguments bind the loop variables at definition time; a closure would
                # capture the variables themselves, and the next subTest would rebind them.
                def sleep(seconds, _delays=delays, _clock=clock):
                    _delays.append(seconds)
                    _clock[0] += seconds

                client = client_module.SpicyClient(
                    "test-key",
                    transport=transport,
                    sleep=sleep,
                    monotonic=lambda _clock=clock: _clock[0],
                    random_value=lambda: 0.5,
                )
                task = client.wait_for_terminal("job-1", timeout_seconds=60.0)
                self.assertEqual(task["state"], terminal)
                self.assertEqual(delays, [2.0, 3.0])

    def test_retry_backoff_and_business_error(self):
        delays = []
        transport = QueueTransport([
            response({"code": 50301, "msg": "busy", "request_id": "req-retry"}, 503),
            response({"code": 200, "msg": "success", "data": {"total": 0, "items": []}}),
        ])
        client = client_module.SpicyClient(
            "test-key",
            transport=transport,
            sleep=delays.append,
            random_value=lambda: 0.5,
        )
        client.list_models()
        self.assertEqual(delays, [0.5])

        failing = client_module.SpicyClient(
            "test-key",
            transport=QueueTransport([
                response({"code": 40303, "msg": "forbidden", "request_id": "req-403"}, 403)
            ]),
            max_retries=0,
        )
        with self.assertRaises(client_module.SpicyApiError) as raised:
            failing.list_models()
        self.assertEqual(raised.exception.status, 403)
        self.assertEqual(raised.exception.code, 40303)
        self.assertEqual(raised.exception.request_id, "req-403")

    def test_request_and_wait_timeouts(self):
        request_client = client_module.SpicyClient(
            "test-key",
            transport=QueueTransport([TimeoutError("timed out")]),
            max_retries=0,
        )
        with self.assertRaises(client_module.SpicyTimeoutError):
            request_client.list_models()

        clock = [0.0]
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": {
                "taskId": "job-1", "model": "provider/model", "state": "running"}})
        ])
        client = client_module.SpicyClient(
            "test-key",
            transport=transport,
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            random_value=lambda: 0.5,
        )
        with self.assertRaises(client_module.SpicyTimeoutError) as raised:
            client.wait_for_terminal("job-1", timeout_seconds=1.0)
        self.assertEqual(raised.exception.task_id, "job-1")


UPLOAD_TICKET = {
    "fileId": "fil_01k3m8x9q2z4v7n5p6r8s0t1w3",
    "key": "spicy://f/fil_01k3m8x9q2z4v7n5p6r8s0t1w3",
    "uploadUrl": "https://storage.example/spicy-inputs/tmp/obj?X-Amz-Signature=deadbeef",
    "method": "PUT",
    "headers": {"Content-Type": "image/png", "Content-Length": "4"},
    "expiresAt": "2026-09-20T09:32:11Z",
    "maxBytes": 10485760,
}
COMMITTED_FILE = {
    "fileId": "fil_01k3m8x9q2z4v7n5p6r8s0t1w3",
    "status": "ready",
    "bytes": 4,
    "contentType": "image/png",
    "sha256": "a" * 64,
    "uri": "spicy://f/fil_01k3m8x9q2z4v7n5p6r8s0t1w3",
    "expiresAt": "2026-09-21T09:12:11Z",
}
PNG_BYTES = b"\x89PNG"

WEBHOOK_SECRET = "whsec-example"
WEBHOOK_TIMESTAMP = "1700000000"
V2_BODY = b'{"code":200,"msg":"success","data":{"taskId":"job-1"},"request_id":"whk-1"}'
# An independently derived constant: common/crypto.SignWebhook on the Go side produces this exact
# string for the same inputs. If the concatenation order, the hex(sha256(body)) layer or the Base64
# encoding drifts anywhere, this goes red.
V2_SIGNATURE = "1ZDMnG7UvpEtxshgOKXHVFR+olokpyj5oEaeZYDJpoY="


def sign(task_id: str, timestamp: str, body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Re-implements the signature from the wording of the contract, without calling the code under
    test, so the two can be compared."""
    digest = hashlib.sha256(body).hexdigest()
    message = f"{task_id}.{timestamp}.{digest}".encode()
    return base64.b64encode(hmac.new(secret.encode(), message, hashlib.sha256).digest()).decode()


def query_of(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


class UploadTests(unittest.TestCase):
    def client(self, transport, **kwargs):
        return client_module.SpicyClient(
            "test-key",
            transport=transport,
            sleep=lambda seconds: None,
            random_value=lambda: 0.5,
            **kwargs,
        )

    def test_upload_runs_three_steps_and_forwards_ticket_headers_verbatim(self):
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": UPLOAD_TICKET}),
            (200, {"ETag": '"abc"'}, b""),
            response({"code": 200, "msg": "success", "data": COMMITTED_FILE}),
        ])
        committed = self.client(transport, max_retries=0).upload_bytes(
            PNG_BYTES, content_type="image/png"
        )

        # The order is fixed: ticket, PUT, commit. Skip a step and there is no usable spicy:// URI.
        self.assertEqual([call[0] for call in transport.calls], ["POST", "PUT", "POST"])
        self.assertTrue(transport.calls[0][1].endswith("/common/upload-url"))
        self.assertEqual(transport.calls[1][1], UPLOAD_TICKET["uploadUrl"])
        self.assertTrue(
            transport.calls[2][1].endswith("/files/fil_01k3m8x9q2z4v7n5p6r8s0t1w3/commit")
        )

        # The declared byte count must equal the real one: a byte either way and commit answers 40003.
        self.assertEqual(
            json.loads(transport.calls[0][3]), {"contentType": "image/png", "bytes": 4}
        )

        # Equality, not containment: Content-Type and Content-Length are both covered by the
        # signature, and sending one too few or one too many earns a 403 that did not come from us.
        self.assertEqual(dict(transport.calls[1][2]), UPLOAD_TICKET["headers"])

        # A signed URL carries its own credentials; adding an Authorization header on top is
        # rejected by some storage implementations.
        self.assertNotIn("Authorization", transport.calls[1][2])

        # The PUT sends the raw bytes themselves - not JSON, not Base64, not multipart.
        self.assertEqual(transport.calls[1][3], PNG_BYTES)

        # Moving bytes uses the upload timeout, not the 30 seconds meant for one API call - 90 MiB
        # does not fit in 30 seconds, and that failure reads like a network problem when the client
        # cut it off itself.
        self.assertEqual(transport.calls[1][4], client_module.UPLOAD_TIMEOUT_SECONDS)
        self.assertLessEqual(transport.calls[2][4], client_module.REQUEST_TIMEOUT_SECONDS)

        # The commit leg, conversely, must carry Authorization: that one is our own endpoint.
        self.assertEqual(transport.calls[2][2]["Authorization"], "Bearer test-key")

        # Only the uri returned by commit may go into input; the key from the ticket is unusable
        # until commit succeeds.
        self.assertEqual(committed["uri"], "spicy://f/fil_01k3m8x9q2z4v7n5p6r8s0t1w3")

    def test_upload_rejects_bad_declaration_before_touching_the_network(self):
        transport = QueueTransport([])
        client = self.client(transport, max_retries=0)

        # The contract accepts exactly these eight; a wrong type means commit reports 40003 only
        # after 90 MiB has already been transferred.
        with self.assertRaises(ValueError):
            client.create_upload_url(content_type="image/svg+xml", byte_count=4)
        # 10 MiB for images and 90 MiB for audio and video are two different lines and cannot share
        # one ceiling.
        with self.assertRaises(ValueError):
            client.create_upload_url(content_type="image/png", byte_count=10 * 1024 * 1024 + 1)
        # The audio and video line is 90 MiB, a different ceiling.
        self.assertEqual(client_module.max_upload_bytes("video/mp4"), 90 * 1024 * 1024)
        self.assertEqual(client_module.max_upload_bytes("image/png"), 10 * 1024 * 1024)
        # What matters is that no request went out at all: these errors are decidable locally,
        # whereas deciding them on the wire means waiting for the whole file to transfer before
        # getting a 40003.
        self.assertEqual(transport.calls, [])

    def test_upload_ticket_is_not_retried(self):
        transport = QueueTransport([
            response({"code": 503, "msg": "busy", "request_id": "req-503"}, 503),
            response({"code": 200, "msg": "success", "data": UPLOAD_TICKET}),
        ])
        client = self.client(transport, max_retries=3)
        with self.assertRaises(client_module.SpicyApiError):
            client.create_upload_url(content_type="image/png", byte_count=4)
        # Every ticket is a write authorisation against object storage, issued under a tight fuse;
        # a retry does not make the previous one usable, it only signs another nobody will use.
        self.assertEqual(len(transport.calls), 1)

    def test_ticket_max_bytes_is_the_authoritative_ceiling(self):
        ticket = {**UPLOAD_TICKET, "maxBytes": 2}
        transport = QueueTransport([response({"code": 200, "msg": "success", "data": ticket})])
        with self.assertRaises(client_module.SpicyUploadError):
            self.client(transport, max_retries=0).upload_bytes(PNG_BYTES, content_type="image/png")
        # A ticket's limit can be stricter than the global one (a model schema may narrow it), and
        # exceeding it means not starting the transfer at all.
        self.assertEqual(len(transport.calls), 1)

    def test_presigned_put_failure_stops_before_commit(self):
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": UPLOAD_TICKET}),
            (403, {}, b"<?xml version=\"1.0\"?><Error><Code>SignatureDoesNotMatch</Code></Error>"),
            response({"code": 200, "msg": "success", "data": COMMITTED_FILE}),
        ])
        client = self.client(transport, max_retries=3)
        with self.assertRaises(client_module.SpicyUploadError) as raised:
            client.upload_bytes(PNG_BYTES, content_type="image/png")
        self.assertEqual(raised.exception.status, 403)
        # Storage answers with XML, not our envelope: parsing it as one reads out as "not valid
        # JSON" and points at a problem that does not exist. And a failed PUT must not be followed
        # by a commit - that only earns a 40003.
        self.assertEqual(len(transport.calls), 2)
        # This leg is not retried automatically: resending the same bytes cannot fix a signature
        # mismatch.
        self.assertEqual(transport.calls[1][0], "PUT")

    def test_upload_file_infers_content_type_from_the_name(self):
        self.assertEqual(client_module.content_type_for_path("/tmp/a.PNG"), "image/png")
        self.assertEqual(client_module.content_type_for_path("clip.mp4"), "video/mp4")
        with self.assertRaises(ValueError):
            client_module.content_type_for_path("notes.txt")


class QuoteAndLifecycleTests(unittest.TestCase):
    def client(self, transport, **kwargs):
        return client_module.SpicyClient(
            "test-key",
            transport=transport,
            sleep=lambda seconds: None,
            random_value=lambda: 0.5,
            **kwargs,
        )

    def test_quote_feeds_create_task(self):
        quote = {
            "quoteId": "eyJ2IjoxfQ.signature",
            "model": "provider/model",
            "estimatedCost": "0.120000000",
            "maxCharge": "0.120000000",
            "currency": "USD",
            "quantity": "1",
            "unit": "per_image",
            "expiresAt": "2026-09-20T09:35:00Z",
        }
        transport = QueueTransport([
            response({"code": 200, "msg": "success", "data": quote}),
            response(
                {
                    "code": 200,
                    "msg": "success",
                    "data": {
                        "taskId": "job-1",
                        "state": "queued",
                        "estimatedCost": "0.120000000",
                        "deadlineAt": "2026-09-20T09:40:00Z",
                    },
                },
                202,
            ),
        ])
        client = self.client(transport, max_retries=0)
        quoted = client.quote_task(model="provider/model", input_data={"prompt": "hello"})
        client.create_task(
            model="provider/model",
            input_data={"prompt": "hello"},
            idempotency_key="logical-request-1",
            quote_id=quoted["quoteId"],
            expected_cost=quoted["estimatedCost"],
        )

        self.assertTrue(transport.calls[0][1].endswith("/jobs/quote"))
        # A quote creates no task and holds no funds, so it must not consume an idempotency key -
        # consuming it would deny that key to the real submission.
        self.assertNotIn("Idempotency-Key", transport.calls[0][2])
        # The quote body is the create-task body, without a self-referential quoteId or
        # expectedCost.
        self.assertEqual(
            json.loads(transport.calls[0][3]),
            {"model": "provider/model", "input": {"prompt": "hello"}},
        )

        body = json.loads(transport.calls[1][3])
        self.assertEqual(body["quoteId"], quoted["quoteId"])
        # The server compares expectedCost digit for digit against the freshly computed price - the
        # number being compared is estimatedCost.
        self.assertEqual(body["expectedCost"], "0.120000000")
        self.assertEqual(transport.calls[1][2]["Idempotency-Key"], "logical-request-1")

    def test_price_changed_is_not_retried(self):
        transport = QueueTransport([
            response(
                {"code": 40901, "msg": "The price changed", "request_id": "req-409"}, 409
            ),
            response({"code": 200, "msg": "success", "data": {"taskId": "job-1"}}, 202),
        ])
        client = self.client(transport, max_retries=3)
        with self.assertRaises(client_module.SpicyApiError) as raised:
            client.create_task(
                model="provider/model",
                input_data={"prompt": "hello"},
                idempotency_key="logical-request-1",
                quote_id="stale.quote",
                expected_cost="0.10",
            )
        # Resending an identical request only earns another 40901: the remedy is a fresh quote, not
        # a backed-off retry.
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(raised.exception.code, 40901)
        self.assertIn("quote again", raised.exception.recovery)
        # Recovery has to reuse the original idempotency key, or the work may be paid for twice.
        self.assertIn("Idempotency-Key", raised.exception.recovery)

    def test_purge_sends_no_idempotency_key(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "taskId": "job-1",
                    "contentState": "purged",
                    "purgedAt": "2026-09-20T10:00:00Z",
                    "contentRemovedBy": "user",
                    "billingRetained": True,
                    "mediaDeletionPending": True,
                },
            }),
        ])
        purged = self.client(transport, max_retries=0).purge_task("job-1")
        self.assertTrue(transport.calls[0][1].endswith("/jobs/purge"))
        # This endpoint never reads Idempotency-Key; the taskId is the idempotency key. Sending one
        # only suggests it is being honoured, and retry logic then gets written against semantics
        # that do not exist.
        self.assertNotIn("Idempotency-Key", transport.calls[0][2])
        self.assertEqual(json.loads(transport.calls[0][3]), {"taskId": "job-1"})
        # What is destroyed is content, not accounting: the ledger, the amount charged, the state
        # and the request_id all stay put.
        self.assertTrue(purged["billingRetained"])

    def test_download_url_and_account_paths(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {"key": "outputs/0", "url": "https://cdn.example/x", "expiresAt": "z"},
            }),
            response({
                "code": 200,
                "msg": "success",
                "data": {"available": "10.00", "held": "0.25", "total": "10.25"},
            }),
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "from": "2026-09-01",
                    "to": "2026-09-08",
                    "currency": "USD",
                    "totalCalls": 3,
                    "totalSpend": "0.36",
                    "days": [],
                    "models": [],
                },
            }),
        ])
        client = self.client(transport, max_retries=0)
        client.create_download_url("job-1", "outputs/0")
        client.get_balance()
        client.get_usage(from_date="2026-09-01", to_date="2026-09-08")

        self.assertTrue(transport.calls[0][1].endswith("/common/download-url"))
        self.assertEqual(
            json.loads(transport.calls[0][3]), {"taskId": "job-1", "key": "outputs/0"}
        )
        self.assertTrue(transport.calls[1][1].endswith("/chat/credit"))
        # from and to are Python keywords, so the parameter names differ - but the names on the wire
        # must not follow them.
        self.assertEqual(
            query_of(transport.calls[2][1]),
            {"from": ["2026-09-01"], "to": ["2026-09-08"]},
        )

    def test_download_url_is_not_retried(self):
        transport = QueueTransport([
            response({"code": 503, "msg": "busy", "request_id": "req-503"}, 503),
            response({"code": 200, "msg": "success",
                      "data": {"key": "k", "url": "u", "expiresAt": "z"}}),
        ])
        client = self.client(transport, max_retries=3)
        with self.assertRaises(client_module.SpicyApiError):
            client.create_download_url("job-1")
        # Same reasoning as issuing a ticket: each call signs another short-lived link, and a retry
        # cannot repair the previous failure.
        self.assertEqual(len(transport.calls), 1)

    def test_task_list_pagination_repeats_every_filter(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "items": [{"taskId": "t1"}],
                    "hasMore": True,
                    "nextCursor": "cursor-2",
                },
            }),
            response({
                "code": 200,
                "msg": "success",
                "data": {"items": [{"taskId": "t2"}], "hasMore": False},
            }),
        ])
        client = self.client(transport, max_retries=0)
        ids = [
            task["taskId"]
            for task in client.iter_tasks(
                from_date="2026-09-01", to_date="2026-09-08", state="succeeded", limit=1
            )
        ]
        self.assertEqual(ids, ["t1", "t2"])

        first, second = query_of(transport.calls[0][1]), query_of(transport.calls[1][1])
        # The first page carries no cursor; every page after it carries the one the previous page
        # returned.
        self.assertNotIn("cursor", first)
        self.assertEqual(second["cursor"], ["cursor-2"])
        # Filters must be repeated verbatim on every page: change one and the cursor no longer
        # points into the same result set.
        for field in ("from", "to", "state", "limit"):
            self.assertEqual(second[field], first[field])
        # Stop when hasMore is false; do not infer the end from an empty items array.
        self.assertEqual(len(transport.calls), 2)


class ResultAndErrorSemanticsTests(unittest.TestCase):
    def client(self, transport, **kwargs):
        return client_module.SpicyClient(
            "test-key",
            transport=transport,
            sleep=lambda seconds: None,
            random_value=lambda: 0.5,
            **kwargs,
        )

    def test_output_helpers_survive_a_result_without_assets(self):
        text_task = {"state": "succeeded", "output": {"text": "transcribed lyrics"}}
        # Some endpoints answer in output.text and carry no assets key whatsoever; reaching for
        # output["assets"][0] raises KeyError on a perfectly successful task.
        self.assertEqual(client_module.output_assets(text_task), [])
        self.assertEqual(client_module.output_text(text_task), "transcribed lyrics")

        file_task = {
            "state": "succeeded",
            "output": {"assets": [{"key": "outputs/0", "url": "https://cdn.example/x.mp4"}]},
        }
        self.assertIsNone(client_module.output_text(file_task))
        self.assertEqual(client_module.output_assets(file_task)[0]["key"], "outputs/0")

        # A failed task has no output at all, and neither helper may blow up on it.
        failed = {"state": "failed", "errorCode": "content_rejected"}
        self.assertEqual(client_module.output_assets(failed), [])
        self.assertIsNone(client_module.output_text(failed))

    def test_wait_keeps_polling_while_assets_are_pending(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "taskId": "job-1",
                    "state": "succeeded",
                    "output": {"assets": [{"key": "outputs/0", "pending": True}]},
                },
            }),
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "taskId": "job-1",
                    "state": "succeeded",
                    "output": {"assets": [{"key": "outputs/0", "url": "https://cdn.example/x"}]},
                },
            }),
        ])
        task = self.client(transport, max_retries=0).wait_for_terminal(
            "job-1", timeout_seconds=60.0
        )
        # Succeeded does not mean the output can be downloaded: a pending asset has no url, and
        # returning then hands the caller a result they cannot fetch.
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(client_module.output_assets(task)[0]["url"], "https://cdn.example/x")

    def test_wait_returns_when_a_pending_asset_is_unavailable(self):
        transport = QueueTransport([
            response({
                "code": 200,
                "msg": "success",
                "data": {
                    "taskId": "job-1",
                    "state": "succeeded",
                    "output": {"assets": [{"key": "o", "pending": True, "unavailable": True}]},
                },
            }),
        ])
        task = self.client(transport, max_retries=0).wait_for_terminal(
            "job-1", timeout_seconds=60.0
        )
        # An asset marked unavailable will never change again, so waiting only burns the local
        # budget down to a timeout - and a timeout means "the remote state is unknown", when here it
        # is perfectly well known.
        self.assertEqual(task["state"], "succeeded")
        self.assertEqual(len(transport.calls), 1)

    def test_each_new_business_code_gets_its_own_handling(self):
        guidance = client_module.RECOVERY_BY_CODE
        # All four remedies differ: merging any two would have one class handled as another.
        self.assertEqual(len({guidance[code] for code in (40003, 40004, 50302, 503)}), 4)
        self.assertIn("upload", guidance[40003])
        self.assertIn("parameter named in the message", guidance[40004])
        self.assertIn("refunded", guidance[50302])
        self.assertIn("retry_after_seconds", guidance[503])

        # 40003 and 40004 are both 400s: resending an identical request yields an identical answer.
        for code in (40003, 40004):
            transport = QueueTransport([
                response({"code": code, "msg": "no", "request_id": "r"}, 400),
                response({"code": 200, "msg": "success", "data": {}}),
            ])
            client = self.client(transport, max_retries=3)
            with self.assertRaises(client_module.SpicyApiError) as raised:
                client.get_balance()
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(guidance[code], raised.exception.recovery)
            self.assertEqual(len(transport.calls), 1)

        # 503, 50301 and 50302 share one HTTP status; only the business code tells them apart.
        transport = QueueTransport([
            response({"code": 503, "msg": "dependency down", "request_id": "r"},
                     503, {"Retry-After": "7"}),
        ])
        client = self.client(transport, max_retries=0)
        with self.assertRaises(client_module.SpicyApiError) as raised:
            client.get_balance()
        self.assertEqual(raised.exception.status, 503)
        self.assertEqual(raised.exception.code, 503)
        # When the server names a delay, honour it rather than overriding it with local back-off.
        self.assertEqual(raised.exception.retry_after_seconds, 7.0)

        transport = QueueTransport([
            response({"code": 50302, "msg": "upstream failed", "request_id": "r"}, 503),
        ])
        with self.assertRaises(client_module.SpicyApiError) as raised:
            self.client(transport, max_retries=0).get_balance()
        self.assertEqual(raised.exception.code, 50302)
        self.assertIn("refunded", raised.exception.recovery)


class WebhookVerificationTests(unittest.TestCase):
    def test_signature_matches_the_server_formula(self):
        computed = client_module.compute_webhook_signature(
            "job-1", WEBHOOK_TIMESTAMP, V2_BODY, WEBHOOK_SECRET
        )
        # Checked twice: once against an independently rewritten formula, once against a fixed
        # constant. The formula alone would pass silently if both sides were changed wrongly
        # together; the constant alone would not show which step went wrong.
        self.assertEqual(computed, sign("job-1", WEBHOOK_TIMESTAMP, V2_BODY))
        self.assertEqual(computed, V2_SIGNATURE)

    def test_v2_delivery_reads_the_task_id_from_data(self):
        verified = client_module.verify_webhook(
            raw_body=V2_BODY,
            timestamp=WEBHOOK_TIMESTAMP,
            signature=V2_SIGNATURE,
            payload_version="2",
            secret=WEBHOOK_SECRET,
            now=lambda: 1700000010,
        )
        self.assertEqual(verified["task_id"], "job-1")
        # In v2 the request_id is a stable delivery id: deliveries are resent, and the receiver
        # needs it to deduplicate.
        self.assertEqual(verified["delivery_id"], "whk-1")
        self.assertEqual(verified["payload"]["data"]["taskId"], "job-1")

    def test_v1_delivery_reads_the_task_id_from_the_top_level(self):
        body = (
            b'{"task_id":"job-1","model":"provider/model","state":"succeeded",'
            b'"cost":"0.1","created_at":"2026-09-20T09:00:00Z"}'
        )
        verified = client_module.verify_webhook(
            raw_body=body,
            timestamp=WEBHOOK_TIMESTAMP,
            signature=sign("job-1", WEBHOOK_TIMESTAMP, body),
            payload_version=1,
            secret=WEBHOOK_SECRET,
            now=lambda: 1700000010,
        )
        self.assertEqual(verified["task_id"], "job-1")
        # v1 has no delivery id, and one must not be invented.
        self.assertIsNone(verified["delivery_id"])

        # Reading a v1 payload the v2 way finds no taskId, and the taskId is part of the signing
        # string: looking in the wrong place surfaces as "signature mismatch" and points at a secret
        # problem that does not exist.
        with self.assertRaises(client_module.SpicyWebhookError) as raised:
            client_module.verify_webhook(
                raw_body=body,
                timestamp=WEBHOOK_TIMESTAMP,
                signature=sign("job-1", WEBHOOK_TIMESTAMP, body),
                payload_version=2,
                secret=WEBHOOK_SECRET,
                now=lambda: 1700000010,
            )
        self.assertEqual(raised.exception.reason, "invalid_payload")

    def test_tampered_body_and_wrong_secret_are_rejected(self):
        tampered = V2_BODY.replace(b'"job-1"', b'"job-2"')
        with self.assertRaises(client_module.SpicyWebhookError) as raised:
            client_module.verify_webhook(
                raw_body=tampered,
                timestamp=WEBHOOK_TIMESTAMP,
                signature=V2_SIGNATURE,
                payload_version=2,
                secret=WEBHOOK_SECRET,
                now=lambda: 1700000010,
            )
        # The signature covers the whole body: one changed byte must fail, or anyone could forge a
        # terminal callback.
        self.assertEqual(raised.exception.reason, "invalid_signature")

        with self.assertRaises(client_module.SpicyWebhookError) as raised:
            client_module.verify_webhook(
                raw_body=V2_BODY,
                timestamp=WEBHOOK_TIMESTAMP,
                signature=V2_SIGNATURE,
                payload_version=2,
                secret="whsec-wrong",
                now=lambda: 1700000010,
            )
        self.assertEqual(raised.exception.reason, "invalid_signature")

    def test_stale_delivery_is_rejected_after_the_signature_checks_out(self):
        with self.assertRaises(client_module.SpicyWebhookError) as raised:
            client_module.verify_webhook(
                raw_body=V2_BODY,
                timestamp=WEBHOOK_TIMESTAMP,
                signature=V2_SIGNATURE,
                payload_version=2,
                secret=WEBHOOK_SECRET,
                now=lambda: 1700000000 + 301,
            )
        # A signature never expires; only the timestamp tolerance stops a captured delivery from
        # being replayed.
        self.assertEqual(raised.exception.reason, "stale_timestamp")

        # The order must not be reversed: a stale payload with a bad signature is reported as a
        # signature failure - an unauthenticated body must not decide anything, including whether it
        # is too old.
        with self.assertRaises(client_module.SpicyWebhookError) as raised:
            client_module.verify_webhook(
                raw_body=V2_BODY,
                timestamp=WEBHOOK_TIMESTAMP,
                signature="AAAA",
                payload_version=2,
                secret=WEBHOOK_SECRET,
                now=lambda: 1700000000 + 301,
            )
        self.assertEqual(raised.exception.reason, "invalid_signature")

    def test_malformed_headers_and_bodies_have_their_own_reasons(self):
        base = {
            "raw_body": V2_BODY,
            "timestamp": WEBHOOK_TIMESTAMP,
            "signature": V2_SIGNATURE,
            "payload_version": 2,
            "secret": WEBHOOK_SECRET,
            "now": lambda: 1700000010,
        }
        for overrides, reason in (
            ({"timestamp": "not-a-number"}, "invalid_timestamp"),
            ({"payload_version": 3}, "invalid_version"),
            ({"secret": "  "}, "invalid_secret"),
            ({"raw_body": b"{not json"}, "invalid_json"),
            ({"raw_body": b'["array"]'}, "invalid_payload"),
            ({"max_body_bytes": 4}, "body_too_large"),
        ):
            with self.subTest(reason=reason):
                with self.assertRaises(client_module.SpicyWebhookError) as raised:
                    client_module.verify_webhook(**{**base, **overrides})
                # Every rejection carries its own reason: collapsed into a single "verification
                # failed", a misconfigured secret and an actual forgery look identical in the logs.
                self.assertEqual(raised.exception.reason, reason)



class PublishReadinessTests(unittest.TestCase):
    """One test each for four defects that only surface at release time.

    What they have in common is that none of them turns an existing test red: an unvalidated
    base_url merely sends the key out, a missing User-Agent merely earns a 403 from edge
    protection, retrying 50302 merely wastes four attempts, and a missed Retry-After merely
    ignores the window the server asked for. Not one of them raises.
    """

    def test_base_url_must_not_leak_the_key_over_plain_http(self):
        # Not validating means the Bearer key travels in the clear, with nothing visible on the
        # caller's side.
        for url, why in [
            ("http://evil.example.com/api/v1", "http on a non-loopback host"),
            ("https://user:pass@api.spicyapi.ai/api/v1", "credentials embedded in the URL"),
            ("https://api.spicyapi.ai/api/v1?token=x", "a query string"),
            ("api.spicyapi.ai", "not an absolute URL"),
        ]:
            with self.subTest(why=why), self.assertRaises(
                ValueError, msg=f"{why} was allowed through"
            ):
                client_module.SpicyClient(api_key="sk-spicy-x", base_url=url)

        # Local development needs http on loopback, which must not be caught by this.
        client_module.SpicyClient(api_key="sk-spicy-x", base_url="http://127.0.0.1:8080/api/v1")

    def test_requests_carry_a_versioned_user_agent(self):
        # Without this header urllib sends its own Python-urllib/3.x. As recorded in
        # scripts/check_contract.py in this repository, the docs site's edge protection answers 403
        # to exactly that.
        seen = {}

        def transport(method, url, headers, body, timeout):
            seen.update(headers)
            return 200, {}, json.dumps(
                {"code": 200, "msg": "ok",
                 "data": {"taskId": "t", "model": "m", "state": "running"}}
            ).encode()

        client_module.SpicyClient(api_key="sk-spicy-x", transport=transport).get_task("tsk_1")
        agent = seen.get("User-Agent")
        self.assertIsNotNone(agent, "the request carried no User-Agent")
        self.assertTrue(agent.startswith("spicyapi-python/"), agent)
        self.assertIn(client_module.__version__, agent,
                      "the version in the UA disagrees with the package version")

    def test_50302_is_not_resent_under_the_same_key(self):
        """50302 rides on a 503, so anything looking only at the status retries it as ordinary
        upstream unavailability.

        The contract says this idempotency key has already recorded the failure, and reusing it
        only replays that failure. Retrying is not fatal - nothing is charged twice - but it makes a
        doomed request take four attempts to report, and the final error reads as "we retried and it
        still failed", burying the actual remedy: a fresh idempotency key. 50301 is the control: that
        one should be retried.
        """
        for code, expected, why in [(50302, 1, "a recorded failure must not be replayed"),
                                    (50301, 4, "ordinary upstream unavailability should be retried")]:
            with self.subTest(code=code):
                attempts = []

                def transport(method, url, headers, body, timeout, _c=code, _seen=attempts):
                    _seen.append(1)
                    return 503, {}, json.dumps(
                        {"code": _c, "msg": "upstream unavailable", "data": None}
                    ).encode()

                client = client_module.SpicyClient(
                    api_key="sk-spicy-x", transport=transport, sleep=lambda _: None
                )
                with self.assertRaises(client_module.SpicyApiError):
                    client.get_task("tsk_1")
                self.assertEqual(len(attempts), expected, why)

    def test_wait_seconds_is_sent_verbatim_and_widens_the_local_timeout(self):
        """Three things about creating a task with wait, each able on its own to make the feature
        fail silently.

        1. The value goes out verbatim: the server does the clamping (anything over 60 becomes 60,
           non-positive integers are ignored), and clamping again locally would only reject a
           working value once the platform relaxes.
        2. The local request timeout has to accommodate the server-side wait - a default 30-second
           request timeout cuts the call off locally while the server is still waiting, which looks
           like "the feature does nothing" when the real cause is two timeouts colliding.
        3. When it is not passed, the parameter must not appear in the query string at all.
        """
        seen = []

        def transport(method, url, headers, body, timeout):
            seen.append((url, timeout))
            return 200, {}, json.dumps(
                {"code": 200, "msg": "ok",
                 "data": {"taskId": "job-1", "state": "queued"}}
            ).encode()

        client = client_module.SpicyClient(api_key="sk-spicy-x", transport=transport)

        client.create_task(model="m", input_data={}, idempotency_key="k1", wait_seconds=45)
        url, timeout = seen[-1]
        self.assertIn("wait=45", url)
        self.assertGreaterEqual(
            timeout, 45,
            "the local request timeout is shorter than the server-side wait budget, so this call "
            "would be cut off while the server is still waiting"
        )

        # Values over 60 go out verbatim too - clamping is the server's job.
        client.create_task(model="m", input_data={}, idempotency_key="k2", wait_seconds=600)
        self.assertIn("wait=600", seen[-1][0])

        # Not passing it means it must not appear.
        client.create_task(model="m", input_data={}, idempotency_key="k3")
        self.assertNotIn("wait=", seen[-1][0])
        # No widening: the client's default request timeout still applies. A range rather than
        # equality, because a real clock is running and min(budget, remaining) always comes out a
        # hair under the budget.
        self.assertLessEqual(seen[-1][1], client.request_timeout_seconds)
        self.assertGreater(seen[-1][1], client.request_timeout_seconds - 1)

        with self.assertRaises(ValueError):
            client.create_task(model="m", input_data={}, idempotency_key="k4", wait_seconds=-1)

    def test_a_waited_creation_is_not_assumed_to_be_finished(self):
        """Once the wait budget runs out what comes back is an acceptance response, identical in
        shape to a terminal record and differing only in state.

        The assumption "I passed wait, so it must be finished" carries a still-running task forward,
        and nothing raises. is_terminal is the test for it.
        """
        def transport(method, url, headers, body, timeout):
            return 200, {}, json.dumps(
                {"code": 200, "msg": "ok",
                 "data": {"taskId": "job-1", "state": "running"}}
            ).encode()

        client = client_module.SpicyClient(api_key="sk-spicy-x", transport=transport)
        task = client.create_task(model="m", input_data={}, idempotency_key="k", wait_seconds=30)
        self.assertFalse(client_module.is_terminal(task),
                         "an acceptance response returned once the budget ran out was read as terminal")
        self.assertTrue(client_module.is_terminal({"state": "succeeded"}))
        self.assertTrue(client_module.is_terminal({"state": "failed"}))

    def test_list_tasks_does_not_cap_the_limit_locally(self):
        """Platform limits are not enforced client-side.

        The day the server relaxes limit from 100 to 200, a hard local ceiling would reject a value
        that already works - and the user only sees the SDK refuse, finds the server perfectly
        willing, and has nothing in between to say who refused. Non-positive values are still
        rejected: that is an unambiguous caller error rather than a platform constraint that can
        expire. The Go client follows the same rule.
        """
        sent = []

        def transport(method, url, headers, body, timeout):
            sent.append(url)
            return 200, {}, json.dumps(
                {"code": 200, "msg": "ok", "data": {"items": [], "hasMore": False}}
            ).encode()

        client = client_module.SpicyClient(api_key="sk-spicy-x", transport=transport)
        client.list_tasks(limit=500)
        self.assertIn("limit=500", sent[-1],
                      "a value the server may well accept was blocked locally")

        with self.assertRaises(ValueError):
            client.list_tasks(limit=0)

    def test_an_empty_filter_is_refused_by_name(self):
        """An empty filter is rejected by name, neither dropped silently nor sent as-is.

        ``None`` means "this filter is not set"; ``""`` means "I computed a filter and it came out
        empty". Dropping it hands the caller the entire unfiltered list with no way to notice;
        sending it earns a 400 from the server that does not name the parameter.
        """
        sent = []

        def transport(method, url, headers, body, timeout):
            sent.append(url)
            return 200, {}, json.dumps(
                {"code": 200, "msg": "ok", "data": {"items": [], "hasMore": False}}
            ).encode()

        client = client_module.SpicyClient(api_key="sk-spicy-x", transport=transport)
        for field in ("state", "model", "cursor"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as caught:
                    client.list_tasks(**{field: ""})
                self.assertIn(field, str(caught.exception),
                              "the error does not name the offending parameter")
        self.assertEqual(sent, [], "a rejected call must not reach the wire")

        # Leaving it unset remains valid: the parameter simply does not appear.
        client.list_tasks()
        self.assertTrue(sent[-1].endswith("/jobs"), sent[-1])

    def test_an_unbounded_response_body_is_refused_instead_of_buffered(self):
        """The other end sends whatever it likes, and read() accepts all of it until the process is
        killed by the OOM reaper.

        Nothing along that path raises - a broken intermediary or a hijacked connection can emit
        bytes indefinitely, and list_models(include_schema=True) is a large response to begin with,
        so nobody would find "this one is unusually big" suspicious.
        """

        class EndlessStream:
            """Supplies whatever is asked for and never ends - what a malicious or broken
            connection looks like."""

            def __init__(self):
                self.calls = 0

            def read(self, size=None):
                self.calls += 1
                if size is None:
                    # The old unbounded path: hand over "everything" in one go.
                    return b"x" * (10 * 1024 * 1024)
                return b"x" * size

        stream = EndlessStream()
        with self.assertRaises(client_module.SpicyApiError) as caught:
            client_module._read_capped(stream)
        self.assertIn("ceiling", str(caught.exception))
        # The ceiling is 8 MiB and each block is 64 KiB, so the number of reads is bounded and
        # predictable.
        self.assertLess(stream.calls, 200,
                        "far more blocks were read than expected; the ceiling did not hold")

        # A response of ordinary size must not be caught by this.
        class ShortStream:
            def __init__(self):
                self.done = False

            def read(self, size=None):
                if self.done:
                    return b""
                self.done = True
                return b'{"code":200}'

        self.assertEqual(client_module._read_capped(ShortStream()), b'{"code":200}')

    def test_retry_after_is_read_case_insensitively(self):
        # HTTP header names are case-insensitive, and urllib preserves whatever casing the server
        # sent. A literal lookup misses a lowercase spelling - back-off then quietly falls back to
        # local exponential timing and stops honouring the server's window.
        self.assertEqual(client_module._header({"retry-after": "7"}, "Retry-After"), "7")
        self.assertEqual(client_module._header({"RETRY-AFTER": "7"}, "Retry-After"), "7")
        self.assertIsNone(client_module._header({"x-other": "7"}, "Retry-After"))

if __name__ == "__main__":
    unittest.main()


class WaitDeadlinePreservesTaskId(unittest.TestCase):
    """When the polling budget runs out, the exception raised must carry the task_id.

    This was found in the Java client on 2026-09-20 and, after checking all four one by one,
    confirmed to be the same defect in Python: the final lap issues a request with a near-zero
    timeout, and when that request times out it raises a plain "request timed out" with no task_id.
    The caller loses the task id at the exact moment it matters most: the task is still running and
    still being billed, while all they hold is "the request timed out", with nothing tying the call
    to the task.

    The TypeScript client does not have this hole: its whole wait shares one abort signal, and that
    signal's reason carries the taskId. This docstring used to claim "Go does not have it either";
    that half was wrong - an inference written without running anything, which an audit report then
    copied forward as "confirmed". Running it showed that Go merely lacked the near-zero-timeout
    cause, while a single polling timeout still ended the entire wait; it was fixed the same day.
    """

    def test_exhausted_budget_still_names_the_task(self):
        clock = [0.0]

        def monotonic() -> float:
            return clock[0]

        def transport(method, url, headers, body, timeout):
            # The point is to simulate a request that genuinely times out. If the client still
            # issues one when the budget is short, the timeout handed down is a tiny value, and this
            # raises - exactly what happens in production.
            if timeout < 1.0:
                raise TimeoutError("socket timed out")
            clock[0] += 4.0
            return response(
                {"code": 200, "msg": "success", "data": {"taskId": "tsk_slow", "state": "running"}}
            )

        client = client_module.SpicyClient(
            api_key="sk-spicy-x", transport=transport, monotonic=monotonic, sleep=lambda _: None
        )
        with self.assertRaises(client_module.SpicyTimeoutError) as caught:
            client.wait_for_terminal("tsk_slow", timeout_seconds=8.5)

        # The assertion that matters is not "a timeout was raised" but "the timeout that was raised
        # carries the task id".
        self.assertEqual(caught.exception.task_id, "tsk_slow")
        self.assertIn("tsk_slow", str(caught.exception))

    def test_a_polling_request_that_times_out_does_not_end_the_wait(self):
        """One polling attempt timing out is not this wait failing, and the timeout that does get
        raised must carry the task_id.

        This pins the shape the test above fails to catch. That test's fake transport only times out
        when ``timeout < 1.0``, so it happens to cover only the case the "less than a second of
        budget" floor already handles. The common case is the other one: 600 seconds of budget, and
        the first poll stalling for the full request_timeout of 30 seconds. Before the fix that
        would
          - give up after a single poll, voiding 600 seconds of budget over one piece of
            turbulence, and
          - raise the timeout from _request, which carries no task_id.
        The task is still running and still being billed at that point, and the caller holds nothing
        that could lead them back to it.
        """
        clock = [0.0]
        calls = []

        def transport(method, url, headers, body, timeout):
            calls.append(timeout)
            clock[0] += timeout  # the request really does stall for the timeout it was given
            raise client_module.SpicyTimeoutError(
                f"request exceeded the local {timeout}s timeout"
            )

        client = client_module.SpicyClient(
            api_key="sk-spicy-x",
            transport=transport,
            monotonic=lambda: clock[0],
            sleep=lambda s: clock.__setitem__(0, clock[0] + s),
        )
        with self.assertRaises(client_module.SpicyTimeoutError) as caught:
            client.wait_for_terminal("tsk_flaky", timeout_seconds=600.0)

        # 1. The attempt that timed out must not end the wait: 600 seconds affords many retries.
        self.assertGreater(
            len(calls), 1,
            "one polling timeout ended the whole wait - while budget remains it must keep polling"
        )
        # 2. The timeout finally raised must carry the task id.
        self.assertEqual(caught.exception.task_id, "tsk_flaky")
        self.assertIn("tsk_flaky", str(caught.exception))
