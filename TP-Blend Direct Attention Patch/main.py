import os
import time
import torch
from PIL import Image
from diffusers import LEditsPPPipelineStableDiffusionXL, DDIMScheduler, AutoencoderKL

if __name__ == '__main__':
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
        model_id,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.float32
    ).to("cuda")

    # Use correct relative path to dataset image inside TP-Blend
    image_path = os.path.join("..", "dataset", "person", "forrest-gump.jpg")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024))

    _ = pipe.invert(image=image, num_inversion_steps=50, skip=0.2)

    editing_prompt = ["Man", "Taylor Swift", "jean shorts", "oil painting"]
    start_time = time.time()
    edited_image = pipe(
        editing_prompt=editing_prompt,
        reverse_editing_direction=[True, False, False, False],
        edit_guidance_scale=[5.0, 10.0, 0, 0],
        edit_threshold=[0, 0, 0, 0],
        edit_warmup_steps=5
    ).images[0]
    end_time = time.time()
    duration = end_time - start_time
    print(f"Image editing process completed in {duration:.2f} seconds.")

    # Save output into TP-Blend/output_images
    output_dir = os.path.join("..", "output_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"edited_{'_'.join(editing_prompt)}.jpg")
    edited_image.save(output_path)
    print(f"Edited image saved to: {output_path}")
