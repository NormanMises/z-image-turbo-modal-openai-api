# Z-Image Turbo API for Modal

[English](README.md) · [简体中文](README.zh-CN.md)

Deploy [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) on [Modal](https://modal.com/) and expose it through an OpenAI-compatible image API.

The service provides:

- `GET /v1/models`
- `POST /v1/images/generations`
- Official Z-Image Native PyTorch inference
- One image per request, returned as Base64 JSON

## Requirements

- A Modal account with GPU access
- Modal CLI

The local Python version only needs to be supported by the Modal CLI. The deployed container uses Python 3.13 internally.

Z-Image-Turbo is public and does not require a Hugging Face token.

## Deploy

Install and authenticate the Modal CLI:

```bash
python -m pip install modal
modal setup
```

Create the Secret used by the deployed function:

```bash
modal secret create z-image-api API_KEY=replace-with-a-long-random-secret
```

The Secret must contain `API_KEY`. Keep the real value out of Git and out of browser code.

`API_KEY` is the key that API clients send in `Authorization: Bearer ...`. It is separate from your Modal login token.

Deploy from the repository root:

```bash
modal deploy z_image_turbo_modal.py
```

Modal builds the runtime image from the `modal.Image` definition in `z_image_turbo_modal.py`. The command prints the application URL. Add `/v1` to that URL to get the API base URL:

```text
https://your-modal-app.modal.run/v1
```

The app stores model weights in the `hf-hub-cache` Modal Volume. The first model load downloads the weights; later starts reuse the Volume.

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
    "model": "z-image-turbo",
    "prompt": "a bold typographic poster",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

The request fields are:

- `model`: `z-image-turbo`
- `prompt`: required text prompt
- `size`: `WIDTHxHEIGHT`, default `1024x1024`
- `response_format`: `b64_json`

The endpoint always generates one image. It does not expose `n`, `steps`, `quality`, or `seed`. Size handling is delegated to the Z-Image Native implementation.

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
    model="z-image-turbo",
    prompt="a bold typographic poster",
    size="1024x1024",
    response_format="b64_json",
)
```

## Image and model weights

The repository defines the Modal Image in Python and pins the Z-Image source dependency to a tested commit. Each Modal workspace builds the image from that definition.

The model weights stay in the Modal Volume rather than in the image. This keeps image builds smaller and lets new containers reuse the cached weights.

For other image workflows, see the [Modal existing images guide](https://modal.com/docs/guide/existing-images).

## Local check

```bash
python -m py_compile z_image_turbo_modal.py
```

## Security

- Store `API_KEY` in the `z-image-api` Modal Secret.
- Never commit `.dev.vars`, Modal tokens, or other credentials.
- Rotate the key if it is exposed.
- CORS currently allows all origins. Restrict `allow_origins` before serving untrusted browser clients.

## Upstream

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Modal documentation](https://modal.com/docs)
