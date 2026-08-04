"""Z-Image 模型的加载器与生成后端。

深 module：base 权重下载/加载、checkpoint（ComfyUI 格式）注入、
原生生成与图片序列化都收敛在这里，对外只有 ``ZImageBackend`` 一个 interface。
"""

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Callable

Z_IMAGE_STEPS = 8
Z_IMAGE_GUIDANCE = 0.0


@dataclass(frozen=True)
class CheckpointSpec:
    repo: str
    filename: str
    fmt: str = "comfyui"


def convert_comfyui_state_dict(state_dict: dict) -> dict:
    """将 ComfyUI fused 命名转换为原生 Z-Image 命名。

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
            raise ValueError(f"意外的 ComfyUI checkpoint key: {key}")

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


_CHECKPOINT_CONVERTERS: dict[str, Callable[[dict], dict]] = {
    "comfyui": convert_comfyui_state_dict,
}


def _image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ZImageBackend:
    """Z-Image 系列模型的加载与生成后端。

    ``checkpoint`` 为 None 时加载原生权重；否则按 ``fmt`` 转换注入。
    """

    def __init__(
        self,
        model_dir: str,
        repo_id: str,
        checkpoint: CheckpointSpec | None = None,
        steps: int = Z_IMAGE_STEPS,
        guidance: float = Z_IMAGE_GUIDANCE,
    ):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from utils import ensure_model_weights, load_from_local_dir, set_attention_backend

        self._steps = steps
        self._guidance = guidance

        print("正在加载 base Z-Image-Turbo 组件...")
        model_path = ensure_model_weights(model_dir, repo_id=repo_id, verify=False)
        self.components = load_from_local_dir(
            model_path,
            device="cuda",
            dtype=torch.bfloat16,
            compile=False,
        )
        attention_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
        set_attention_backend(attention_backend)

        if checkpoint is not None:
            converter = _CHECKPOINT_CONVERTERS.get(checkpoint.fmt)
            if converter is None:
                raise ValueError(f"不支持的 checkpoint 格式: {checkpoint.fmt}")
            print(f"正在注入 checkpoint: {checkpoint.repo}/{checkpoint.filename} ({checkpoint.fmt})")
            checkpoint_path = hf_hub_download(repo_id=checkpoint.repo, filename=checkpoint.filename)
            state_dict = load_file(checkpoint_path, device="cpu")
            converted = converter(state_dict)
            del state_dict
            transformer = self.components["transformer"]
            transformer.load_state_dict(converted, strict=True)
            del converted

        torch.cuda.empty_cache()
        print(f"Z-Image native model loaded; attention={attention_backend}")
        print("模型加载完成！")

    def generate(self, prompt: str, width: int, height: int) -> list[str]:
        import torch
        from zimage import generate as native_generate

        images = native_generate(
            prompt=prompt,
            **self.components,
            height=height,
            width=width,
            num_inference_steps=self._steps,
            guidance_scale=self._guidance,
        )
        torch.cuda.empty_cache()
        return [_image_to_base64(image_obj) for image_obj in images]
