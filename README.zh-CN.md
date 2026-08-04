# 在 Modal 上部署图片生成 API

[English](README.md) · [简体中文](README.zh-CN.md)

在 [Modal](https://modal.com/) 上部署一个统一的、兼容 OpenAI Images API 的图片生成接口，内置三个自托管模型：

| `model` id | 模型 | 后端 |
|---|---|---|
| `z-image-turbo` | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Z-Image 原生 PyTorch |
| `dbz8-sda` | [Dark Beast DBZiT8 SDA@FOK](https://civarchive.com/models/2242173?modelVersionId=2774410) | Z-Image Turbo 微调（原生） |
| `flux.2-klein-9b` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | Diffusers 管线 |

服务提供：

- `GET /v1/models` —— 列出全部三个模型
- `POST /v1/images/generations` —— 每次请求生成一张图片，以 Base64 JSON 返回
- 通过请求里的 `model` 字段分发到对应模型（必填）
- OpenAI 标准错误格式：`{"error": {"message", "type", "param", "code"}}`

## 部署要求

- 已开通 GPU 的 Modal 账号
- Modal CLI

本地 Python 版本只需要满足 Modal CLI 的要求。部署到 Modal 的远端运行镜像内部使用 Python 3.13（Z-Image 模型）和 3.12（FLUX.2 Klein 9B）。

Z-Image-Turbo 及其微调模型是公开模型，不需要 Hugging Face token。FLUX.2 Klein 9B 是 gated 模型，需要 `flux2-hf` Secret 里的 HF token。

## 部署

安装并登录 Modal CLI：

```bash
python -m pip install modal
modal setup
```

创建 API 使用的 Secret：

```bash
modal secret create z-image-api API_KEY=替换为一个较长的随机密钥
```

该 Secret 必须包含 `API_KEY`。不要把真实密钥提交到 Git，也不要写进浏览器端代码。

`API_KEY` 就是客户端调用本 API 时放在 `Authorization: Bearer ...` 中的 key，与 Modal 登录 token 是两回事。

如果使用 FLUX.2 模型，还需要创建存 HF token 的 Secret：

```bash
modal secret create flux2-hf HF_TOKEN=替换为你的-hf-token
```

在仓库根目录执行部署：

```bash
modal deploy images_app.py
```

Modal 会根据 `images_app.py` 中的 `modal.Image` 定义构建运行镜像，并输出应用地址。给这个地址加上 `/v1`，就是 API Base URL：

```text
https://your-modal-app.modal.run/v1
```

模型权重保存在 Modal Volume 中（Z-Image 系列用 `hf-hub-cache`，FLUX.2 用 `flux2-klein-9b-cache`）。第一次加载模型时会下载权重，之后的新实例会复用 Volume 中的文件。

## 项目结构

- `images_api.py` —— 共享的 OpenAI 兼容 web 接口（鉴权、OpenAI 标准错误、`/v1/models` 和生成路由）。`create_images_app(models, metas)` 按 `model` 字段分发到对应模型。
- `images_app.py` —— 统一部署入口。定义三个模型类，各自挂不同的 image / volume / secret。

## 认证

使用标准 OpenAI 请求头发送 Secret 中的值：

```http
Authorization: Bearer your-api-key
```

## API 调用

先设置地址和 API key：

```bash
export BASE_URL="https://your-modal-app.modal.run/v1"
export API_KEY="your-api-key"
```

获取模型列表：

```bash
curl "$BASE_URL/models" \
  -H "Authorization: Bearer $API_KEY"
```

生成图片：

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

请求字段：

- `model`：必填，取 `z-image-turbo` / `dbz8-sda` / `flux.2-klein-9b` 之一
- `prompt`：必填的文本提示词
- `size`：`WIDTHxHEIGHT`，默认 `1024x1024`
- `response_format`：只支持 `b64_json`

每次请求只生成一张图片，不提供 `n`、`steps`、`quality` 或 `seed`。尺寸处理交给各模型的推理实现。

错误遵循 OpenAI 格式：400 为 `invalid_request_error`，401 为 `authentication_error`，500 为 `server_error`。

也可以使用 OpenAI Python 客户端：

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

## 镜像和模型权重

仓库在 Python 文件中定义 Modal Image，并将 Z-Image 源码依赖固定到经过验证的提交。每个 Modal workspace 都会根据这份定义构建运行镜像。

模型权重保存在 Modal Volume 中，不打包进镜像。这样镜像更小，新实例也能复用已经下载的权重。

其他镜像用法参见 [Modal Existing Images 官方文档](https://modal.com/docs/guide/existing-images)。

## 本地检查

```bash
python -m py_compile images_api.py images_app.py
```

## 安全

- 将 `API_KEY` 保存在 `z-image-api` Modal Secret 中。
- 不要提交 `.env`、`.dev.vars`、Modal token 或其他凭据。
- 如果 key 泄露，请立即轮换。
- 当前 CORS 允许所有来源。面向不受信任的浏览器客户端时，请限制 `allow_origins`。

## 上游项目

- [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [Z-Image-Turbo 模型](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- [Modal 文档](https://modal.com/docs)
