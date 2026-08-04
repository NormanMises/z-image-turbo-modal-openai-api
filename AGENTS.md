# Repository Guidelines

## 项目结构
一个 Modal App（`images_app.py`）里一个 FastAPI 应用 + 三个生图模型端点，共用 OpenAI Images 兼容接口。

- `images_api.py`：共享的 OpenAI Images 兼容 web 接口（鉴权、OpenAI 标准错误、`/v1/models` 与生成路由）。`create_images_app(models, metas)` 按请求里的 `model` 字段分发到对应模型。
- `zimage_loader.py`：Z-Image 系列模型的加载与生成后端（`ZImageBackend`），含 ComfyUI checkpoint → 原生权重的转换（`CheckpointSpec`）。
- `images_app.py`：统一部署入口。Z-Image 家族由 `Z_IMAGE_MODELS` spec 表驱动（工厂按 spec 生成 cls）；FLUX.2 保持显式 cls。
- 真实密钥放在 Modal Secret 中，不要提交。
- `public/`、`functions/`、Cloudflare Pages 前端和旧 R2 图库链路已移除。

## 构建、调试与部署
- Python / Modal 命令先进入 `deep` conda/mamba 环境，执行 Python 一律用 `mamba run -n deep python -u ...`。
- 语法检查：`python -m py_compile images_api.py images_app.py zimage_loader.py`
- 部署：`modal deploy images_app.py`
- 生成接口：`POST /v1/images/generations`；模型列表：`GET /v1/models`

## 环境变量与 Secret
- 客户端使用 `Authorization: Bearer <API_KEY>`。
- Modal Secret `z-image-api` 只需包含 `API_KEY`。
- Z-Image-Turbo 权重不是 gated，无需 HF token。

## API 约定
`POST /v1/images/generations` 接收 OpenAI Images 风格 JSON：

```json
{
  "model": "z-image-turbo",
  "prompt": "a bold typographic poster",
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

- `model` 为必填，取值见 `GET /v1/models`（`z-image-turbo` / `dbz8-sda` / `dbz6` / `flux.2-klein-9b`）。
- 每个请求固定生成一张图片；不传 `n`、`steps`、`quality` 或 `seed`。
- `size` 为 `WIDTHxHEIGHT`，默认 `1024x1024`，尺寸校验由各模型推理函数处理。
- `response_format` 只支持 `"b64_json"`，返回图片 Base64。
- 错误响应为 OpenAI 标准格式：`{"error": {"message", "type", "param", "code"}}`；401 为 `authentication_error`，400 为 `invalid_request_error`，500 为 `server_error`。

## 代码风格
尽量做最小改动，保持文件局部风格一致。Python 使用 `snake_case`，常量使用全大写。新增调试优先用 `ic(...)`，不要扩散 `print(...)`。

## 测试要求
仓库目前没有独立自动化测试；改动后至少运行：

- `python -m py_compile images_api.py images_app.py`
- 如涉及 `images_api.py`：跑本地 TestClient 验证鉴权、错误格式与路由（含 `model` 分发）。
- 如涉及部署：`modal deploy images_app.py`
- 如涉及接口行为：用最小尺寸请求做 smoke test。

## 安全与许可
不要提交 `.env`、`.dev.vars`、HF token、Modal token 或任何密钥。
