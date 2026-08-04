"""统一的 Modal 部署：一个 App、一个 FastAPI 应用、三个生图模型端点。

- ``images_api.create_images_app`` 提供共享的 OpenAI Images 兼容 web 接口。
- 三个模型各自是一个 ``@app.cls``，按请求里的 ``model`` 字段分发。
- 部署：``modal deploy images_app.py``
"""

import base64
import io
import os
import re

import modal

from images_api import ImagesMeta, create_images_app


CACHE_DIR = "/cache"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_COMMIT = "26f23eda626ffadda020b04ff79488e1d72004cd"
Z_IMAGE_DIR = f"{CACHE_DIR}/Z-Image-Turbo"
Z_IMAGE_STEPS = 8
DBZ_REPO = "GuangyuanSD/REDCraft-DarkBeast-Z-Image-TURBO"
DBZ_FILENAME = "DarkBeast-ZImageTurbo/DarkBeastZ8-SDA@Fok-BF16-ComfyUI.safetensors"
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
)


def image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def convert_dbz8_state_dict(state_dict: dict) -> dict:
    """将 DBZiT8 的 ComfyUI fused 命名转换为原生 Z-Image 命名。

    转换规则（已与 Tongyi-MAI/Z-Image-Turbo 原生权重逐 key 核对）：
      - 去掉 ``model.diffusion_model.`` 前缀
      - ``attention.qkv`` 融合矩阵按 3 等分拆成 to_q/to_k/to_v
      - ``q_norm``/``k_norm`` -> ``norm_q``/``norm_k``；``attention.out`` -> ``attention.to_out.0``
      - ``x_embedder.*`` -> ``all_x_embedder.2-1.*``
      - ``final_layer.*`` -> ``all_final_layer.2-1.*``
    """
    qkv_re = re.compile(r"^(.*\.attention)\.qkv\.weight$")
    result = {}
    for key, tensor in state_dict.items():
        if key == "__metadata__":
            continue
        if not key.startswith("model.diffusion_model."):
            raise ValueError(f"意外的 DBZiT8 checkpoint key: {key}")

        native = key[len("model.diffusion_model."):]

        qkv_match = qkv_re.match(native)
        if qkv_match:
            chunk = tensor.shape[0] // 3
            result[f"{qkv_match.group(1)}.to_q.weight"] = tensor[:chunk]
            result[f"{qkv_match.group(1)}.to_k.weight"] = tensor[chunk:2 * chunk]
            result[f"{qkv_match.group(1)}.to_v.weight"] = tensor[2 * chunk:]
            continue

        native = native.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
        native = native.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
        native = native.replace(".attention.out.weight", ".attention.to_out.0.weight")

        if native == "x_embedder.weight":
            native = "all_x_embedder.2-1.weight"
        elif native == "x_embedder.bias":
            native = "all_x_embedder.2-1.bias"
        elif native == "final_layer.adaLN_modulation.1.weight":
            native = "all_final_layer.2-1.adaLN_modulation.1.weight"
        elif native == "final_layer.adaLN_modulation.1.bias":
            native = "all_final_layer.2-1.adaLN_modulation.1.bias"
        elif native == "final_layer.linear.weight":
            native = "all_final_layer.2-1.linear.weight"
        elif native == "final_layer.linear.bias":
            native = "all_final_layer.2-1.linear.bias"

        result[native] = tensor
    return result


@app.cls(
    image=z_image_image,
    gpu="L40S",
    volumes={CACHE_DIR: z_image_cache},
)
class ZImageModel:
    @modal.enter()
    def load_model(self):
        import torch
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        print("正在加载 Z-Image-Turbo 模型...")
        model_path = ensure_model_weights(Z_IMAGE_DIR, repo_id=Z_IMAGE_REPO, verify=False)
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
            num_inference_steps=Z_IMAGE_STEPS,
            guidance_scale=0.0,
        )
        torch.cuda.empty_cache()
        return [image_to_base64(image_obj) for image_obj in images]


@app.cls(
    image=z_image_image,
    gpu="L40S",
    volumes={CACHE_DIR: z_image_cache},
)
class DBZ8Model:
    @modal.enter()
    def load_model(self):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        print("正在加载 base Z-Image-Turbo 组件...")
        model_path = ensure_model_weights(Z_IMAGE_DIR, repo_id=Z_IMAGE_REPO, verify=False)
        self.components = load_from_local_dir(
            model_path,
            device="cuda",
            dtype=torch.bfloat16,
            compile=False,
        )
        attention_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        set_attention_backend(attention_backend)

        print("正在下载 DBZiT8 SDA@FOK checkpoint...")
        dbz_path = hf_hub_download(repo_id=DBZ_REPO, filename=DBZ_FILENAME)
        print(f"DBZiT8 checkpoint: {dbz_path}")

        dbz_sd = load_file(dbz_path, device="cpu")
        converted = convert_dbz8_state_dict(dbz_sd)
        del dbz_sd

        transformer = self.components["transformer"]
        transformer.load_state_dict(converted, strict=True)
        del converted
        torch.cuda.empty_cache()
        print(f"DBZiT8 SDA@FOK native model loaded; attention={attention_backend}")
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
            num_inference_steps=Z_IMAGE_STEPS,
            guidance_scale=0.0,
        )
        torch.cuda.empty_cache()
        return [image_to_base64(image_obj) for image_obj in images]


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
        "z-image-turbo": lambda prompt, width, height: ZImageModel().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
        "dbz8-sda": lambda prompt, width, height: DBZ8Model().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
        "flux.2-klein-9b": lambda prompt, width, height: Flux2Klein9B().generate.remote.aio(
            prompt=prompt, width=width, height=height
        ),
    },
    metas={
        "z-image-turbo": ImagesMeta(model="z-image-turbo", owned_by="Tongyi-MAI"),
        "dbz8-sda": ImagesMeta(model="dbz8-sda", owned_by="AiMetatron"),
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
