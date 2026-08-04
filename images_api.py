"""共享的 OpenAI Images 兼容 web 接口。

统一的深 module：鉴权、size 解析、OpenAI 标准错误、
`/v1/models` 与 `POST /v1/images/generations` 路由只在这里实现一次。

``create_images_app`` 接收一个模型注册表（model_id -> generate adapter）和
元数据表（model_id -> ImagesMeta），按请求里的 ``model`` 字段分发到对应模型。
"""

import hmac
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

DEFAULT_SIZE = "1024x1024"

# generate 契约：异步，输入 prompt + 宽高，返回一张图的 b64 列表。
Generate = Callable[[str, int, int], Awaitable[list[str]]]


@dataclass(frozen=True)
class ImagesMeta:
    model: str
    owned_by: str
    default_size: str = DEFAULT_SIZE


class OpenAIError(Exception):
    """OpenAI 标准错误：``{"error": {"message", "type", "param", "code"}}``。"""

    def __init__(self, status_code: int, type_: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {
            "error": {
                "message": message,
                "type": type_,
                "param": param,
                "code": None,
            }
        }


def invalid_request(message: str, param: str | None = None) -> OpenAIError:
    return OpenAIError(400, "invalid_request_error", message, param)


def authentication_error(message: str) -> OpenAIError:
    return OpenAIError(401, "authentication_error", message)


def server_error(message: str) -> OpenAIError:
    return OpenAIError(500, "server_error", message)


def verify_api_key(authorization: str | None):
    expected_secret = os.environ.get("API_KEY")
    if not expected_secret:
        raise server_error("The API key is not configured on the server.")

    scheme, separator, token = (authorization or "").partition(" ")
    provided_secret = token.strip() if separator and scheme.lower() == "bearer" else None

    if not provided_secret or not hmac.compare_digest(provided_secret, expected_secret):
        raise authentication_error("Invalid API key.")


def parse_size(size: object, default_size: str) -> tuple[int, int]:
    if size is None:
        size = default_size
    if not isinstance(size, str):
        raise invalid_request("size must use WIDTHxHEIGHT format.", param="size")

    try:
        raw_width, raw_height = size.lower().split("x", 1)
        width = int(raw_width)
        height = int(raw_height)
    except ValueError as exc:
        raise invalid_request("size must use WIDTHxHEIGHT format.", param="size") from exc

    return width, height


async def read_json_request(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        raise invalid_request("The request body must be a JSON object.") from exc
    if not isinstance(payload, dict):
        raise invalid_request("The request body must be a JSON object.")
    return payload


def build_image_response(images_b64: list[str]) -> JSONResponse:
    created = int(time.time())
    data = [{"b64_json": image_b64} for image_b64 in images_b64]
    return JSONResponse(content={"created": created, "data": data}, headers={"cache-control": "no-store"})


def create_images_app(models: dict[str, Generate], metas: dict[str, ImagesMeta]) -> FastAPI:
    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.exception_handler(OpenAIError)
    async def _handle_openai_error(request: Request, exc: OpenAIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    @web_app.get("/v1/models")
    def list_models(
        authorization: str | None = Header(None, alias="Authorization"),
    ):
        verify_api_key(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": meta.model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": meta.owned_by,
                }
                for meta in metas.values()
            ],
        }

    @web_app.post("/v1/images/generations")
    async def openai_images_generations(
        request: Request,
        authorization: str | None = Header(None, alias="Authorization"),
    ):
        verify_api_key(authorization)
        payload = await read_json_request(request)

        model_id = payload.get("model")
        if not isinstance(model_id, str) or model_id not in models:
            raise invalid_request(
                "model must be one of: " + ", ".join(sorted(models)), param="model"
            )

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise invalid_request("prompt must be a non-empty string.", param="prompt")

        width, height = parse_size(payload.get("size"), metas[model_id].default_size)

        response_format = str(payload.get("response_format") or "b64_json")
        if response_format != "b64_json":
            raise invalid_request("response_format only supports b64_json.", param="response_format")

        try:
            images_b64 = await models[model_id](prompt.strip(), width, height)
        except Exception as exc:
            print(f"image generation failed: {exc!r}")
            raise server_error("Image generation failed.") from exc

        return build_image_response(images_b64)

    return web_app
