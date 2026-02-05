from typing import Optional, Dict

from diffusers import DiffusionPipeline  # type: ignore
from .attn_caof_sasf import CAOFSASFProcessor


def install_tpblend_attention_on_unet(
    pipe: DiffusionPipeline,
    enable_caof: bool = True,
    enable_sasf: bool = True,
    sasf_mode: str = "DSIN",  # "AdaIN" or "DSIN"
    use_ot: bool = True,
    per_block_overrides: Optional[Dict[str, Dict]] = None,
) -> None:
    """
    Replaces ALL attention processors in pipe.unet with CAOFSASFProcessor.
    Call AFTER pipe.invert(...) and BEFORE pipe(...).
    """
    unet = pipe.unet
    attn_procs = {}

    per_block_overrides = per_block_overrides or {}

    for name in unet.attn_processors.keys():
        cfg = {
            "enable_CAOF": enable_caof,
            "enable_SASF": enable_sasf,
            "sasf_mode": sasf_mode,
            "use_ot": use_ot,
        }
        for prefix, override in per_block_overrides.items():
            if name.startswith(prefix):
                cfg.update(override)

        attn_procs[name] = CAOFSASFProcessor(
            enable_CAOF=cfg["enable_CAOF"],
            enable_SASF=cfg["enable_SASF"],
            sasf_mode=cfg["sasf_mode"],
            use_ot=cfg["use_ot"],
        )

    unet.set_attn_processor(attn_procs)
