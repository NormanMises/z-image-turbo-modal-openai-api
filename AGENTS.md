# Repository Guidelines

## 项目结构
本仓库现在只保留 Modal 自托管 Z-Image Turbo 图片生成 API。

- `z_image_turbo_modal.py`：Modal + FastAPI 后端，部署 Tongyi-MAI/Z-Image-Turbo，并提供 OpenAI Images API 兼容端点。
- 真实密钥放在 Modal Secret 中，不要提交。
- `public/`、`functions/`、Cloudflare Pages 前端和旧 R2 图库链路已移除。

## 构建、调试与部署
- Python / Modal 命令先进入 `deep` conda/mamba 环境。
- 语法检查：`python -m py_compile z_image_turbo_modal.py`
- 部署：`modal deploy z_image_turbo_modal.py`
- 线上健康检查：`GET /`
- 生成接口：`POST /v1/images/generations`

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

- 每个请求固定生成一张图片；不传 `n`、`steps`、`quality` 或 `seed`。
- `size` 为 `WIDTHxHEIGHT`，默认 `1024x1024`，尺寸校验由官方 Z-Image Native 推理函数处理。
- `response_format` 只支持 `"b64_json"`，返回图片 Base64。

## 代码风格
尽量做最小改动，保持文件局部风格一致。Python 使用 `snake_case`，常量使用全大写。新增调试优先用 `ic(...)`，不要扩散 `print(...)`。

## 测试要求
仓库目前没有独立自动化测试；改动后至少运行：

- `python -m py_compile z_image_turbo_modal.py`
- 如涉及部署：`modal deploy z_image_turbo_modal.py`
- 如涉及接口行为：用最小尺寸请求做 smoke test。

## 安全与许可
不要提交 `.env`、`.dev.vars`、HF token、Modal token 或任何密钥。
