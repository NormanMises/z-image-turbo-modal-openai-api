import base64
import hmac
import io
import os
import time

import modal
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


APP_NAME = "flux2-klein-9b-api"
CACHE_DIR = "/cache"
MODEL_REPO = "black-forest-labs/FLUX.2-klein-9B"
MODEL_ID = "flux.2-klein-9b"
DEFAULT_SIZE = "1024x1024"


web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("flux2-klein-9b-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "fastapi[standard]",
        "huggingface_hub[hf_xet]>=1.23.0",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=5.9.0",
        "accelerate==1.12.0",
        "safetensors>=0.8.0",
        "git+https://github.com/huggingface/diffusers.git",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def verify_api_key(authorization: str | None):
    expected_secret = os.environ.get("API_KEY")
    if not expected_secret:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")

    scheme, separator, token = (authorization or "").partition(" ")
    provided_secret = token.strip() if separator and scheme.lower() == "bearer" else None

    if not provided_secret or not hmac.compare_digest(provided_secret, expected_secret):
        raise HTTPException(status_code=401, detail="Unauthorized")


def parse_size(size: object) -> tuple[int, int]:
    if size is None:
        size = DEFAULT_SIZE
    if not isinstance(size, str):
        raise HTTPException(status_code=400, detail="size must use WIDTHxHEIGHT format")

    try:
        raw_width, raw_height = size.lower().split("x", 1)
        width = int(raw_width)
        height = int(raw_height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="size must use WIDTHxHEIGHT format") from exc

    return width, height


def image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.cls(
    image=image,
    gpu="L40S",
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("flux2-hf")],
)
@modal.concurrent(max_inputs=1)
class Flux2Klein9B:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to("cuda")
        cache_volume.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> str:
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=4,
        ).images[0]
        return image_to_base64(image)


async def read_json_request(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload


@web_app.get("/")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "default_size": DEFAULT_SIZE,
        "endpoints": ["/v1/models", "/v1/images/generations"],
    }


@web_app.get("/v1/models")
def list_models(
    authorization: str | None = Header(None, alias="Authorization"),
):
    verify_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "black-forest-labs",
            }
        ],
    }


@web_app.post("/v1/images/generations")
async def openai_images_generations(
    request: Request,
    authorization: str | None = Header(None, alias="Authorization"),
):
    verify_api_key(authorization)
    payload = await read_json_request(request)

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be a non-empty string")

    width, height = parse_size(payload.get("size"))
    response_format = str(payload.get("response_format") or "b64_json")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail="Only b64_json is supported")

    flux = Flux2Klein9B()
    image_b64 = await flux.generate.remote.aio(
        prompt=prompt.strip(),
        width=width,
        height=height,
    )
    return JSONResponse(
        content={"created": int(time.time()), "data": [{"b64_json": image_b64}]},
        headers={"cache-control": "no-store"},
    )


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("z-image-api")],
)
@modal.asgi_app()
def fastapi_app():
    return web_app
