import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
import os


st.set_page_config(page_title="🎨 Image Generator", page_icon="✨", layout="centered")
st.title("🎨 AI Image Generator")
st.write("Type a prompt to generate an image using Stable Diffusion — no API key needed!")

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# ----------------------------
# USER INPUT
# ----------------------------
prompt = st.text_input("🖊️ Enter a prompt:", placeholder="e.g., a cat making pizza")

generate_btn = st.button("✨ Generate Image")

if generate_btn:
    if prompt.strip():
        with st.spinner("Generating your image... ⏳"):
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
st.caption("Made with 💫 Streamlit + Stable Diffusion")
