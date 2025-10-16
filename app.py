import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="CPU AI Image Generator", page_icon="✨", layout="centered")
st.title("🎨 CPU-Friendly AI Image Generator")
st.write("Generates 3 small images per prompt using a lightweight Stable Diffusion model on CPU. No API key needed!")

# LOAD MODEL (CPU, small)
# ----------------------------
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # CPU-friendly
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# ----------------------------
# USER INPUT
# ----------------------------
prompt = st.text_input("Enter your prompt:", placeholder="e.g., A cat making pizza")

size = (256, 256)  # small size for CPU stability

if st.button("Generate 3 Images"):
    if prompt.strip():
        with st.spinner("Generating images on CPU (may take ~20–40 seconds)..."):
            os.makedirs("img", exist_ok=True)
            images = [pipe(prompt, height=size[0], width=size[1]).images[0] for _ in range(3)]

            st.image(images, caption=[f"Result {i+1}" for i in range(3)], use_container_width=True)

            for i, img in enumerate(images):
                filename = f"img/{prompt.replace(' ','_')}_{i+1}.png"
                img.save(filename)
            st.success("✅ Images saved in the `img` folder!")
    else:
        st.warning("Please enter a prompt first.")

st.markdown("---")
st.caption("Made with 💫 Streamlit + CPU-friendly Stable Diffusion")
