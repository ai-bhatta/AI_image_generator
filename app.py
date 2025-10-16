import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="🎨 AI Image Generator", page_icon="✨", layout="centered")
st.title("🎨 CPU-Friendly AI Image Generator")
st.write("Generate images from text using a small Stable Diffusion model (CPU-compatible). No API keys needed!")

# LOAD MODEL (CPU)
# ----------------------------
@st.cache_resource
def load_model():
    model_id = "stabilityai/stable-diffusion-2-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")
    return pipe

pipe = load_model()

# USER INPUT
# ----------------------------
prompt = st.text_input("🖊️ Enter your prompt:", placeholder="e.g., A cat making pizza")

if st.button("✨ Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image (CPU, may take ~30-60 seconds)... ⏳"):
            image = pipe(prompt).images[0]
            st.image(image, caption=f"Generated: {prompt}", use_container_width=True)

            # Save image
            os.makedirs("img", exist_ok=True)
            file_path = f"img/{prompt.replace(' ', '_')}.png"
            image.save(file_path)
            st.success(f"✅ Image saved as: {file_path}")
    else:
        st.warning("Please enter a prompt first!")

st.markdown("---")
st.caption("Made with 💫 Streamlit + CPU-compatible Stable Diffusion")
