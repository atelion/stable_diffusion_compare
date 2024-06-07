from diffusers import StableDiffusionPipeline
import torch
import ImageReward as RM

scoring_model = RM.load("ImageReward-v1.0")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "A bustling farm market overflowing with fresh fall produce on a sunny day"
for idx in range(8):
    images = pipe(prompt).images
    # images[0].save(f"{idx}-{prompt}.png")
    score = scoring_model.score(prompt, images)
    print(score)