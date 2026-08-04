# Modal Image API

[English](README.md) · [简体中文](README.zh-CN.md)

Deploy a unified OpenAI-compatible image API on [Modal](https://modal.com/) serving three self-hosted image generation models:

| `model` id | Model | Backend |
|---|---|---|
| `z-image-turbo` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Z-Image native PyTorch |
| `dbz8-sda` | [Dark Beast DBZiT8 SDA@FOK](https://civarchive.com/models/2242173?modelVersionId=2774410) | Z-Image Turbo fine-tune (native) |
| `flux.2-klein-9b` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Diffusers pipeline |

The service provides:

- `GET /v1/models` — lists all three models
- `POST /v1/images/generations` — generates one image per request, returned as Base64 JSON
- Model dispatch via the required `model` request field
- OpenAI-standard error format: `{"error": {"message", "type", "param", "code"}}`

## Requirements

- A Modal account with GPU access
- Modal CLI

The local Python version only needs to be supported by the Modal CLI. The deployed containers use Python 3.13 (Z-Image models) and 3.12 (FLUX.2 Klein 9B) internally.

Z-Image-Turbo and its fine-tunes are public and do not require a Hugging Face token. FLUX.2 Klein 9B is gated and needs the `flux2-hf` Secret containing your HF token.

## Deploy

Install and authenticate the Modal CLI:

```bash
python -m pip install modal
modal setup
```

Create the Secret used by the API:

```bash
modal secret create z-image-api API_KEY=replace-with-a-long-random-secret
```

The Secret must contain `API_KEY`. Keep the real value out of Git and out of browser code.

`API_KEY` is the key that API clients send in `Authorization: Bearer ...`. It is separate from your Modal login token.

If you use the FLUX.2 model, also create a Secret for the HF token:

```bash
modal secret create flux2-hf HF_TOKEN=replace-with-your-hf-token
```

Deploy from the repository root:

```bash
modal deploy images_app.py
```

Modal builds the runtime images from the `modal.Image` definitions in `images_app.py` and prints the application URL. Add `/v1` to that URL to get the API base URL:

```text
https://your-modal-app.modal.run/v1
```

The app stores model weights in Modal Volumes (`hf-hub-cache` for the Z-Image family, `flux2-klein-9b-cache` for FLUX.2). The first model load downloads the weights; later starts reuse the Volume.

## Structure

- `images_api.py` — shared OpenAI-compatible web interface (auth, OpenAI-standard errors, `/v1/models` and the generation route). `create_images_app(models, metas)` dispatches to a model by the `model` field.
- `images_app.py` — the unified deployment entry. Defines the three model classes, each with its own image / volume / secret.

## Authentication

Send the Secret value in the standard OpenAI header:

```http
Authorization: Bearer your-api-key
```

## API

Set the URL and key in your shell:

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

List available models:

```bash
curl "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY"
```

Generate an image:

```bash
curl "$BASE_URL/images/generations" \
  -X POST \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dbz8-sda",
    "prompt": "a bold typographic poster",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

The request fields are:

- `model`: required, one of `z-image-turbo`, `dbz8-sda`, `flux.2-klein-9b`
- `prompt`: required text prompt
- `size`: `WIDTHxHEIGHT`, default `1024x1024`
- `response_format`: `b64_json`

The endpoint always generates one image. It does not expose `n`, `steps`, `quality`, or `seed`. Size handling is delegated to each model's inference implementation.

Errors follow the OpenAI format; status codes are 400 `invalid_request_error`, 401 `authentication_error`, 500 `server_error`.

Use the OpenAI Python client if you prefer an SDK:

```bash
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-modal-app.modal.run/v1",
)

result = client.images.generate(
    model="dbz8-sda",
    prompt="a bold typographic poster",
    size="1024x1024",
    response_format="b64_json",
)
```

## Image and model weights

The repository defines the Modal Images in Python and pins the Z-Image source dependency to a tested commit. Each Modal workspace builds the images from those definitions.

The model weights stay in the Modal Volumes rather than in the images. This keeps image builds smaller and lets new containers reuse the cached weights.

For other image workflows, see the [Modal existing images guide](https://modal.com/docs/guide/existing-images).

## Local check

```bash
python -m py_compile images_api.py images_app.py
```

## Security

- Store `API_KEY` in the `z-image-api` Modal Secret.
- Never commit `.env`, `.dev.vars`, Modal tokens, or other credentials.
- Rotate the key if it is exposed.
- CORS currently allows all origins. Restrict `allow_origins` before serving untrusted browser clients.

## Upstream

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [Modal documentation](https://modal.com/docs)
