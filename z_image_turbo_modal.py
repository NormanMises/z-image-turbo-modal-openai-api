import base64
import hmac
import io
import os
import time

import modal
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


APP_NAME = "z-image-turbo-api"
CACHE_DIR = "/cache"
MODEL_Z_IMAGE_TURBO = "z-image-turbo"
MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_COMMIT = "26f23eda626ffadda020b04ff79488e1d72004cd"
DEFAULT_SIZE = "1024x1024"
DEFAULT_STEPS = 8
MODEL_DIR = f"{CACHE_DIR}/Z-Image-Turbo"


web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "fastapi[standard]",
        "tqdm",
        f"git+https://github.com/Tongyi-MAI/Z-Image.git@{Z_IMAGE_COMMIT}",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ZIMAGE_ATTENTION": "_native_flash",
        }
    )
)


def verify_api_key(authorization: str | None):
    expected_secret = os.environ.get("API_KEY")
    if not expected_secret:
        raise HTTPException(status_code=500, detail="未配置 API_KEY。")

    scheme, separator, token = (authorization or "").partition(" ")
    provided_secret = token.strip() if separator and scheme.lower() == "bearer" else None

    if not provided_secret or not hmac.compare_digest(provided_secret, expected_secret):
        raise HTTPException(status_code=401, detail="未授权的请求来源。")


def parse_size(size: object) -> tuple[int, int]:
    if size is None:
        size = DEFAULT_SIZE
    if not isinstance(size, str):
        raise HTTPException(status_code=400, detail="size 必须是 WIDTHxHEIGHT 格式。")

    try:
        raw_width, raw_height = size.lower().split("x", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="size 必须是 WIDTHxHEIGHT 格式。") from exc
    try:
        width = int(raw_width)
        height = int(raw_height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="size 必须是 WIDTHxHEIGHT 格式。") from exc

    return width, height


@app.cls(
    image=image,
    gpu="L40S",
    volumes={CACHE_DIR: cache_volume},
)
class ZImageModel:
    @modal.enter()
    def load_model(self):
        import torch
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        print("正在加载 Z-Image-Turbo 模型...")
        model_path = ensure_model_weights(MODEL_DIR, repo_id=MODEL_REPO, verify=False)
        self.components = load_from_local_dir(
            model_path,
            device="cuda",
            dtype=torch.bfloat16,
            compile=False,
        )
        attention_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        set_attention_backend(attention_backend)
        print(f"Z-Image-Turbo native model loaded; attention={attention_backend}")
        print("模型加载完成！")

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        import torch
        from zimage import generate as native_generate

        images = native_generate(
            prompt=prompt,
            **self.components,
            height=height,
            width=width,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=0.0,
        )

        torch.cuda.empty_cache()

        images_b64 = []
        for image_obj in images:
            buffer = io.BytesIO()
            image_obj.save(buffer, format="PNG")
            images_b64.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return images_b64


async def read_json_request(request: Request) -> dict:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象。")
    return payload


def build_image_response(images_b64: list[str], response_format: str):
    created = int(time.time())
    data = [{"b64_json": image_b64} for image_b64 in images_b64]
    return JSONResponse(content={"created": created, "data": data}, headers={"cache-control": "no-store"})


@web_app.get("/")
def health():
    return {
        "ok": True,
        "model": MODEL_Z_IMAGE_TURBO,
        "default_size": DEFAULT_SIZE,
        "default_steps": DEFAULT_STEPS,
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
                "id": MODEL_Z_IMAGE_TURBO,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "Tongyi-MAI",
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
        raise HTTPException(status_code=400, detail="prompt 为必填字符串。")

    width, height = parse_size(payload.get("size"))
    response_format = str(payload.get("response_format") or "b64_json")
    if response_format != "b64_json":
        raise HTTPException(status_code=400, detail="response_format 只支持 b64_json。")
    z_image = ZImageModel()
    images_b64 = await z_image.generate.remote.aio(
        prompt=prompt.strip(),
        width=width,
        height=height,
    )
    return build_image_response(images_b64, response_format)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("z-image-api")],
)
@modal.asgi_app()
def fastapi_app():
    return web_app
