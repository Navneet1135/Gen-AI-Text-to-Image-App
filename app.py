from diffusers import StableDiffusionPipeline
import torch
import gradio as gr
import tempfile
from PIL import Image

model_id = "runwayml/stable-diffusion-v1-5"
token = "YOUR-TOKEN-NO"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=token)
pipe.to("cuda")

def genrate_image(prompt):
  image = pipe(prompt).images[0]
  return image

  interface =gr.Interface(fn = genrate_image,
                        inputs = gr.Textbox(label ="Enter Image Description"),
                         outputs = gr.Image(label="Generated image"),
                        title = "Gen AI Text-to Image App",
                        description = "Enter a decriptive text to generate an image.")

interface.launch(share = True , debug = True)
