import argparse
import json
import os
import time
from typing import Optional, Tuple, List, Any

import torch
from PIL import Image

from diffusers import (
    LEditsPPPipelineStableDiffusionXL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from tp_blend import install_tpblend_attention_on_unet


def load_image(path: str, size_hw: Optional[Tuple[int, int]] = (1024, 1024)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size_hw is not None:
        img = img.resize(size_hw)
    return img


def to_torch_dtype(name: Optional[str]) -> torch.dtype:
    if name is None:
        return torch.float32
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def build_scheduler(model_id: str, name: str):
    name = (name or "ddim").lower()
    if name == "ddim":
        return DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif name in ("dpmpp_2m_sde", "dpmsolverpp", "dpmpp_sde_2m"):
        sched = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
        sched.config.algorithm_type = "sde-dpmsolver++"
        sched.config.solver_order = 2
        return sched
    else:
        return DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_tuple2(v: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if v is None:
        return default
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    return default


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config.json (JSON with // or /* */ comments OK)")
    return ap.parse_args()


def load_jsonc(path: str) -> Any:
    """
    Minimal JSON-with-comments loader:
      - strips // line comments and /* block comments */
      - preserves strings and escapes correctly
    """
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()

    out = []
    i = 0
    n = len(s)
    in_string = False
    string_quote = ""
    in_line_comment = False
    in_block_comment = False

    while i < n:
        ch = s[i]

        # End of line comment
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        # End of block comment
        if in_block_comment:
            if ch == "*" and i + 1 < n and s[i + 1] == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        # Inside string
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                # escape next char
                out.append(s[i + 1])
                i += 2
                continue
            if ch == string_quote:
                in_string = False
            i += 1
            continue

        # Not in string/comment: check comment starts
        if ch == "/" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "/":
                in_line_comment = True
                i += 2
                continue
            elif nxt == "*":
                in_block_comment = True
                i += 2
                continue

        # Start of string
        if ch == '"' or ch == "'":
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue

        # Normal char
        out.append(ch)
        i += 1

    cleaned = "".join(out)
    return json.loads(cleaned)


def main():
    args = parse_args()
    cfg = load_jsonc(args.config)

    # ---------- Model / runtime ----------
    model_id = cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")
    vae_id = cfg.get("vae_id", "madebyollin/sdxl-vae-fp16-fix")
    device = cfg.get("device", "cuda")
    torch_dtype_name = cfg.get("torch_dtype", "float32")
    torch_dtype = to_torch_dtype(torch_dtype_name)

    scheduler_name = cfg.get("scheduler", "ddim")

    # ---------- I/O ----------
    image_path = cfg.get("image_path")
    if not image_path:
        raise ValueError("config.json must set 'image_path'.")

    image_size = as_tuple2(cfg.get("image_size"), (1024, 1024))

    output_dir = cfg.get("output_dir", "./output_images")
    ensure_dir(output_dir)

    # ---------- Inversion ----------
    inv = cfg.get("invert", {})
    num_inversion_steps = int(inv.get("num_inversion_steps", 50))
    skip = float(inv.get("skip", 0.2))

    # ---------- Attention install ----------
    attn = cfg.get("attention", {})
    enable_caof = bool(attn.get("enable_caof", True))
    enable_sasf = bool(attn.get("enable_sasf", True))
    sasf_mode = str(attn.get("sasf_mode", "DSIN"))
    use_ot = bool(attn.get("use_ot", True))
    per_block_overrides = attn.get("per_block_overrides", None)

    # ---------- Edit params ----------
    edit = cfg.get("edit", {})
    editing_prompt = edit.get("editing_prompt", ["Man", "Taylor Swift", "jean shorts", ""])
    reverse_editing_direction = edit.get("reverse_editing_direction", [True, False, False, False])
    edit_guidance_scale = edit.get("edit_guidance_scale", [5.0, 10.0, 0, 0])
    edit_threshold = edit.get("edit_threshold", [0, 0, 0, 0])
    edit_warmup_steps = int(edit.get("edit_warmup_steps", 5))
    edit_cooldown_steps = edit.get("edit_cooldown_steps", None)

    # ---------- Load pipeline ----------
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype)
    scheduler = build_scheduler(model_id, scheduler_name)

    pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
        model_id,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    ).to(device)

    # ---------- Image ----------
    image = load_image(image_path, image_size)

    # 1) Invert (resets attn processors internally)
    _ = pipe.invert(image=image, num_inversion_steps=num_inversion_steps, skip=skip)

    # 2) Install your CAOF+SASF processors on all UNet attention layers
    install_tpblend_attention_on_unet(
        pipe,
        enable_caof=enable_caof,
        enable_sasf=enable_sasf,
        sasf_mode=sasf_mode,
        use_ot=use_ot,
        per_block_overrides=per_block_overrides,
    )

    # 3) Edit
    start_time = time.time()
    edited = pipe(
        editing_prompt=editing_prompt,
        reverse_editing_direction=reverse_editing_direction,
        edit_guidance_scale=edit_guidance_scale,
        edit_threshold=edit_threshold,
        edit_warmup_steps=edit_warmup_steps,
        edit_cooldown_steps=edit_cooldown_steps,
    ).images[0]
    duration = time.time() - start_time
    print(f"Image editing process completed in {duration:.2f} seconds.")

    out_name = f"edited_{'_'.join([str(x) for x in editing_prompt])}.jpg"
    save_path = os.path.join(output_dir, out_name)
    edited.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
