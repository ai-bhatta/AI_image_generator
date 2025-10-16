import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

st.set_page_config(page_title="Lightweight AI Image Generator", layout="centered")
st.title("🎨 CPU-Friendly AI Image Generator")
st.write("Generates images using a small Stable Diffusion model on CPU (no API key)")

@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-base"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32  # CPU-friendly
    )
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

prompt = st.text_input("Enter a prompt:", placeholder="e.g., A cute robot painting a sunset")

if st.button("Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image (CPU, may take ~30 sec)..."):
            image = pipe(prompt).images[0]
            st.image(image, caption=prompt, use_container_width=True)

            os.makedirs("img", exist_ok=True)
            filename = f"img/{prompt.replace(' ','_')}.png"
            image.save(filename)
            st.success(f"Image saved: {filename}")
    else:
        st.warning("Please enter a prompt.")
