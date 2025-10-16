import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

st.set_page_config(page_title="Render AI Image Generator", layout="centered")
st.title("🎨 CPU-Friendly AI Image Generator on Render")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

prompt = st.text_input("Enter your prompt:")
size = (256, 256)

if st.button("Generate 3 Images"):
    if prompt.strip():
        with st.spinner("Generating images..."):
            os.makedirs("img", exist_ok=True)
            images = [pipe(prompt, height=size[0], width=size[1]).images[0] for _ in range(3)]
            st.image(images, caption=[f"Result {i+1}" for i in range(3)], use_container_width=True)
            for i, img in enumerate(images):
                filename = f"img/{prompt.replace(' ','_')}_{i+1}.png"
                img.save(filename)
            st.success("Images saved locally!")
    else:
        st.warning("Enter a prompt first.")
