# TP-Blend: Implementation Code

Implementation code for the paper **“TP-Blend: Textual-Prompt Attention Pairing for Precise Object-Style Blending in Diffusion Models.”**

📄 **Paper landing page (GitHub Pages):** https://felixxinjin1.github.io/TP-Blend/  
📄 **PDF (GitHub Pages):** https://felixxinjin1.github.io/TP-Blend/paper/tp-blend.pdf  
📄 **Paper (OpenReview):** https://openreview.net/forum?id=q6M73uOBZE  
📄 **Paper (OpenReview PDF):** https://openreview.net/pdf?id=q6M73uOBZE  
📄 **arXiv (Abstract):** https://arxiv.org/abs/2601.08011  
📄 **arXiv (PDF):** https://arxiv.org/pdf/2601.08011

**TP-Blend** provides a custom **attention processor** that enables precise object blending and faithful style transfer in diffusion models:
- **SASF — Self-Attention Style Fusion** injects style within self-attention while preserving structure and texture.
- **CAOF — Cross-Attention Object Fusion** uses (optional) optimal transport on cross-attention to map source object features to target regions while preserving background.

---

## 🚀 Two Ways to Run TP-Blend

You can use TP-Blend in **either** of the following ways. Choose the path that fits your workflow.

### **A) Standard Drop‑in Module (no library edits)**
Install TP-Blend’s processor at runtime; your `diffusers` package remains untouched.

**Folder layout**
```
TP-Blend/
├─ tp_blend/
│  ├─ attn_caof_sasf.py       # TP-Blend attention processor (CAOF + SASF)
│  ├─ installer.py            # Installs the processor onto the UNet
│  └─ __init__.py
├─ config.json
├─ environment.tp-blend.yaml
└─ main.py                    # Reads config.json and runs the pipeline
```

**Run**
```bash
# 1) Create & activate the environment
conda env create -f environment.tp-blend.yaml
conda activate TP-Blend

# 2) Launch (reads config.json)
python main.py --config config.json
```
**What this does**
1) Loads model & VAE 
2) Inverts the input image 
3) Installs TP‑Blend attention on UNet layers (external install, no file patching) 
4) Applies edits per `edit.*` in `config.json` 
5) Saves results to `./output_images`

---

### **B) TP-Blend Direct Attention Patch (inline replace in `diffusers`)**
Replace `diffusers/models/attention_processor.py` with the patched file that already includes TP‑Blend logic. A minimal `main.py` is provided for quick tests.

**Folder layout (this repo)**
```
TP-Blend/
└─ TP-Blend Direct Attention Patch/
   ├─ attention_processor.py   # Drop-in replacement for diffusers' file
   └─ main.py                  # Minimal runner using the patched file
```

**Where to copy the patched file** 
Typical Conda path (your Python minor version may vary):
```
/home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py
```
If your environment uses e.g. Python **3.10**, adjust the `python3.8` segment accordingly.

**Steps**
```bash
# 0) Activate the env
conda activate TP-Blend

# 1) (Recommended) Back up the original file
cp -v \
  /home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py \
  /home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py.bak

# 2) Copy in the TP‑Blend patched file from this repo
cp -v "TP-Blend Direct Attention Patch/attention_processor.py" \
  /home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py

# 3) (First time) Create an output folder
mkdir -p ./output_images

# 4) Run the minimal script
python "TP-Blend Direct Attention Patch/main.py"
```
**Revert** at any time:
```bash
mv -v \
  /home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py.bak \
  /home/adm1/anaconda3/envs/TP-Blend/lib/python3.8/site-packages/diffusers/models/attention_processor.py
```
**Notes**
- `TP-Blend Direct Attention Patch/main.py` contains fields like `image_path`, `editing_prompt`, etc. Edit them to match your inputs and experiments.
- Updating `diffusers` (e.g., `pip install -U diffusers`) may overwrite your patch; just re-apply step **2**.
- If your site‑packages path is different (virtualenv, different OS, different Python minor), update the path accordingly.

---

## 1) Environment Setup

Use the provided conda environment file to create an environment named `TP-Blend`.

**File:** `environment.tp-blend.yaml`
```bash
conda env create -f environment.tp-blend.yaml
conda activate TP-Blend
```
- **GPU users:** the file pins PyTorch + CUDA 12.1 via `pytorch-cuda=12.1`.
- **CPU-only:** remove the `pytorch-cuda=12.1` line before creating the env.

---

## 2) Configuration (for Way A: Drop‑in Module)

All runtime options are read from `config.json`. 
**Comments** (`// ...` and `/* ... */`) are supported and stripped at load time.

**Example: `config.json`**
```jsonc
{
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "vae_id": "madebyollin/sdxl-vae-fp16-fix",
  "device": "cuda",
  "torch_dtype": "float32",
  "scheduler": "ddim",

  "image_path": "./dataset/person/forrest-gump.jpg",
  "image_size": [1024, 1024],
  "output_dir": "./output_images",

  "invert": {
    "num_inversion_steps": 50,
    "skip": 0.2
  },

  "attention": {
    "enable_caof": true,
    "enable_sasf": true,
    "sasf_mode": "DSIN",
    "use_ot": true,
    "per_block_overrides": {
      // example: "down_blocks.0": { "enable_CAOF": false }
    }
  },

  "edit": {
    "editing_prompt": ["Man", "Taylor Swift", "jean shorts+white shirt", "oil painting"],
    "reverse_editing_direction": [true, false, false, false],
    "edit_guidance_scale": [5.0, 10.0, 0, 0],
    "edit_threshold": [0, 0, 0, 0],
    "edit_warmup_steps": 5,
    "edit_cooldown_steps": null
  }
}
```

**Tips**
- `attention.sasf_mode`: `"AdaIN"` for classic AdaIN; `"DSIN"` injects more high‑frequency style detail.
- `attention.use_ot`: `true` for OT‑based CAOF; `false` for simple index‑aligned α‑blend ablation.
- `per_block_overrides`: selectively enable/disable CAOF/SASF per UNet block by name prefix.

---

## 3) Run (Way A)

```bash
python main.py --config config.json
```
This will:
1. Load model & VAE 
2. Invert the input image 
3. Install TP‑Blend attention on **all** UNet attention layers (external, no library edits) 
4. Perform editing per `edit.*` settings 
5. Save results to `output_dir`

---

## 4) Run (Way B: TP‑Blend Direct Attention Patch)

After copying `TP-Blend Direct Attention Patch/attention_processor.py` over your environment’s
`diffusers/models/attention_processor.py`, run:

```bash
conda activate TP-Blend
mkdir -p ./output_images
python "TP-Blend Direct Attention Patch/main.py"
```

**Edit inputs in** `TP-Blend Direct Attention Patch/main.py`:
```python
image_path = "/path/to/your/image.jpg"
editing_prompt = ["Man", "Taylor Swift", "jean shorts", "oil painting"]
# reverse_editing_direction, edit_guidance_scale, thresholds, warmup_steps, etc.
```
The script saves a result like:
```
./output_images/edited_Man_Taylor Swift_jean shorts_oil painting.jpg
```

---

## 5) Repository Layout (essentials)

```
tp_blend/
  ├─ attn_caof_sasf.py         # TP‑Blend attention processor (CAOF + SASF)
  ├─ installer.py              # Helper to install the processor onto the UNet
  └─ __init__.py               # Exports installer and processor

TP-Blend Direct Attention Patch/
  ├─ attention_processor.py    # Inline replacement for diffusers/models/attention_processor.py
  └─ main.py                   # Minimal runner using the inline patch

main.py                        # Entry point for Way A (reads config.json)
environment.tp-blend.yaml      # Conda environment file
config.json                    # Runtime configuration (comments supported)
```

---

## 6) Troubleshooting

- **Import/Path errors**: Confirm your actual site‑packages path. If you’re not on Python 3.8, replace `python3.8` in the path with your version (e.g., `python3.10`). 
- **Permission denied**: You may need to run the copy with appropriate permissions or adjust your environment’s path. 
- **Patch “disappears”** after upgrading `diffusers`: Re‑apply the copy step. 
- **CUDA errors**: Ensure the PyTorch/CUDA versions in `environment.tp-blend.yaml` match your driver and hardware.

---

## 7) Citation

If you use this code, please cite:

@article{
jin2025tpblend,
title={{TP}-Blend: Textual-Prompt Attention Pairing for Precise Object-Style Blending in Diffusion Models},
author={Xin Jin and Yichuan Zhong and Yapeng Tian},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=q6M73uOBZE},
note={}
}

---

## 8) License

This project is licensed under the **Apache License 2.0**. 
You may use, modify, and distribute this code under the terms of that license. 
See: http://www.apache.org/licenses/LICENSE-2.0
