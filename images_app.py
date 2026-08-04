"""统一的 Modal 部署：一个 App、一个 FastAPI 应用、多个生图模型端点。

- ``images_api.create_images_app`` 提供共享的 OpenAI Images 兼容 web 接口。
- ``zimage_loader.ZImageBackend`` 封装 Z-Image 系列的加载/生成（含 ComfyUI checkpoint 注入）。
- Z-Image 家族由 ``Z_IMAGE_MODELS`` spec 表驱动，工厂按 spec 生成各模型 cls；
  FLUX.2 是异类（不同 image/volume/后端），保持显式 cls。
- 部署：``modal deploy images_app.py``
"""

import base64
import io
from dataclasses import dataclass

import modal

from images_api import ImagesMeta, create_images_app
from zimage_loader import Z_IMAGE_STEPS, CheckpointSpec, ZImageBackend


CACHE_DIR = "/cache"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_COMMIT = "26f23eda626ffadda020b04ff79488e1d72004cd"
Z_IMAGE_DIR = f"{CACHE_DIR}/Z-Image-Turbo"
DBZ_REPO = "GuangyuanSD/REDCraft-DarkBeast-Z-Image-TURBO"
DBZ_FILENAME = "DarkBeast-ZImageTurbo/DarkBeastZ8-SDA@Fok-BF16-ComfyUI.safetensors"
DBZ6_FILENAME = "DarkBeast-ZImageTurbo/DarkBeastZ6-BlitZ-BF16-ComfyUI.safetensors"
FLUX2_REPO = "black-forest-labs/FLUX.2-klein-9B"
FLUX2_STEPS = 4


app = modal.App("images-api")

z_image_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
flux2_cache = modal.Volume.from_name("flux2-klein-9b-cache", create_if_missing=True)

z_image_image = (
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
    .add_local_file("images_api.py", remote_path="/root/images_api.py")
    .add_local_file("zimage_loader.py", remote_path="/root/zimage_loader.py")
)

flux2_image = (
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
    .add_local_file("images_api.py", remote_path="/root/images_api.py")
    .add_local_file("zimage_loader.py", remote_path="/root/zimage_loader.py")
)


@dataclass(frozen=True)
class ZModelSpec:
    id: str
    owned_by: str
    checkpoint: CheckpointSpec | None = None
    steps: int = Z_IMAGE_STEPS


Z_IMAGE_MODELS = [
    ZModelSpec("z-image-turbo", "Tongyi-MAI"),
    ZModelSpec("dbz8-sda", "AiMetatron", CheckpointSpec(DBZ_REPO, DBZ_FILENAME)),
    ZModelSpec("dbz6", "AiMetatron", CheckpointSpec(DBZ_REPO, DBZ6_FILENAME)),
]


def make_zimage_cls(spec: ZModelSpec):
    """按 spec 动态生成一个 Z-Image 模型 cls（每个一个独立 GPU 容器）。"""
    name = "".join(p.capitalize() for p in spec.id.split("-")) + "Model"

    @modal.enter()
    def load_model(self):
        self.backend = ZImageBackend(
            model_dir=Z_IMAGE_DIR,
            repo_id=Z_IMAGE_REPO,
            checkpoint=spec.checkpoint,
            steps=spec.steps,
        )

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        return self.backend.generate(prompt, width, height)

    cls = type(name, (), {"load_model": load_model, "generate": generate})
    cls = app.cls(image=z_image_image, gpu="L40S", volumes={CACHE_DIR: z_image_cache})(cls)
    globals()[name] = cls  # Modal 运行时要按名字从模块里取 cls
    return cls


Z_IMAGE_CLS = {spec.id: make_zimage_cls(spec) for spec in Z_IMAGE_MODELS}


def image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.cls(
    image=flux2_image,
    gpu="L40S",
    volumes={CACHE_DIR: flux2_cache},
    secrets=[modal.Secret.from_name("flux2-hf")],
)
@modal.concurrent(max_inputs=1)
class Flux2Klein9B:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(
            FLUX2_REPO,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to("cuda")
        flux2_cache.commit()

    @modal.method()
    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=1.0,
            num_inference_steps=FLUX2_STEPS,
        ).images[0]
        return [image_to_base64(image)]


web_app = create_images_app(
    models={
        **{
            mid: (lambda prompt, width, height, cls=cls: cls().generate.remote.aio(
                prompt=prompt, width=width, height=height
            ))
            for mid, cls in Z_IMAGE_CLS.items()
        },
        "flux.2-klein-9b": lambda prompt, width, height: Flux2Klein9B().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
    },
    metas={
        **{spec.id: ImagesMeta(model=spec.id, owned_by=spec.owned_by) for spec in Z_IMAGE_MODELS},
        "flux.2-klein-9b": ImagesMeta(model="flux.2-klein-9b", owned_by="black-forest-labs"),
    },
)


@app.function(
    image=z_image_image,
    secrets=[modal.Secret.from_name("z-image-api")],
)
@modal.asgi_app()
def fastapi_app():
    return web_app
