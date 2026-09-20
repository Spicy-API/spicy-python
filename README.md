<div align="center">

# spicyapi

**Official Python SDK for [SpicyAPI](https://spicyapi.ai)** — image, video and text models behind one API.

[Get a key](https://spicyapi.ai) · [Models](https://spicyapi.ai/models) · [Docs](https://docs.spicyapi.ai) · [Status](https://status.spicyapi.ai)

</div>

---

One endpoint in front of 83 model families across 121 callable endpoints, billed in USD per request
rather than in credits. Media generation is asynchronous and quotable before you spend; text models
speak the OpenAI, Anthropic and Gemini wire formats.

```bash
pip install spicyapi
```

Requires Python 3.11+. **No runtime dependencies**: this package goes into your dependency tree, and
every constraint it adds is one more chance of a conflict.

## Generate something

```python
import os, uuid
from spicyapi import SpicyClient, output_assets

client = SpicyClient()  # reads SPICY_API_KEY from the environment

model = client.get_model("MODEL_ID_FROM_CATALOG")     # copy a real id from list_models()
quote = client.quote_task(model=model["model"], input_data={"prompt": "a lantern in fog"})
print(quote["estimatedCost"], quote["maxCharge"])      # decide before you spend

task = client.create_task(
    model=model["model"],
    input_data={"prompt": "a lantern in fog"},
    idempotency_key=str(uuid.uuid4()),
    quote_id=quote["quoteId"],
    expected_cost=quote["estimatedCost"],
)
final = client.wait_for_terminal(task["taskId"])
for asset in output_assets(final):                     # module-level helper, not a method
    print(asset["url"])
```

Build the `input` from that model's own `inputSchema` — every model has different fields, and
`list_models(include_schema=True)` returns them.

## Start from a local file

Image-to-video, face swap and image editing all need your material on our side first. Upload returns
a `spicy://` URI; that is what goes into `input`.

```python
uploaded = client.upload_file("/path/to/reference.png")
task = client.create_task(
    model="MODEL_ID_FROM_CATALOG",
    input_data={"image": uploaded["uri"], "prompt": "slow dolly in"},
    idempotency_key=str(uuid.uuid4()),
)
```

## Webhooks

`verify_webhook` is a module-level function, so a request handler can use it without building a
client. It compares in constant time and checks the timestamp only after the signature is valid.

```python
from spicyapi import verify_webhook

delivery = verify_webhook(
    raw_body=request.body,                      # the exact bytes, before any parsing
    signature=request.headers["X-Webhook-Signature"],
    timestamp=request.headers["X-Webhook-Timestamp"],
    payload_version=request.headers["X-Webhook-Payload-Version"],
    secret=os.environ["SPICY_WEBHOOK_SECRET"],
)
```

Verify the raw bytes. Re-serialising the parsed JSON changes them, and the signature will never match.

## Two things that will save you money

**Keep one idempotency key per submission.** Reuse it for every resend of that submission, including
after a timeout or a dropped connection. A lost response does not prove the task was not created — a
fresh key turns an unknown outcome into a second paid task.

**A task that succeeds is charged, even if the result disappoints.** Quote first when the price
matters; `quote_task` reserves nothing.

## What this package does not do

**Text models.** They speak the OpenAI, Anthropic and Google Gemini wire formats, so the official
libraries for those already work — point them at `https://api.spicyapi.ai/v1` with the same key.
Wrapping them here would add nothing.

**Browser, mobile and desktop apps.** Never ship this key inside an application: a key compiled into
a client is a public key. Call from your server, or put
`@spicyapi/proxy` in front.

## Errors

Every failure raises `SpicyApiError` with `status`, `code`, `request_id` and `retry_after_seconds`.
Branch on `code`, never on the message text — messages are translated, codes are not.

`503` is shared by three different business codes, so reading the HTTP status alone is not enough:

| code | meaning | what to do |
| --- | --- | --- |
| `40003` | uploaded bytes do not match their ticket | upload again |
| `40004` | no deployment serves that parameter combination | change the parameter named in the message |
| `40901` | the price moved before the task was created | quote again, keep the same idempotency key |
| `503` | a dependency is briefly unavailable | back off by `Retry-After` |
| `50301` | the model has no usable deployment or price right now | do not hammer; refresh the catalogue |
| `50302` | a synchronous generation failed upstream and was refunded | retry with a **new** idempotency key |

`err.recovery` carries the same guidance at runtime.

## Links

- [Documentation](https://docs.spicyapi.ai)
- [API reference](https://docs.spicyapi.ai/docs/api-reference)
- [Source](https://github.com/Spicy-API/spicy-python)

---

<div align="center">
<sub>

Also available in [TypeScript](https://github.com/Spicy-API/spicy-sdk) · **Python** · [Go](https://github.com/Spicy-API/spicy-go) · [PHP](https://github.com/Spicy-API/spicy-php) · [Java](https://github.com/Spicy-API/spicy-java)

</sub>
</div>
