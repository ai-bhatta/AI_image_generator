import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="🎨 AI Image Generator", page_icon="✨", layout="centered")

st.title("🎨 AI Image Generator (Stable Diffusion)")
st.write("Type a prompt and I’ll generate an image using open-source AI!")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# --- User Input ---
prompt = st.text_input("🖊️ Enter your image prompt:", placeholder="e.g. A futuristic city in clouds")

# --- Generate Button ---
if st.button("✨ Generate Image"):
    if prompt.strip():
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption=f"🖼️ {prompt}", use_container_width=True)
            # Save image
            image.save("generated_image.png")
            st.success("✅ Image generated and saved as generated_image.png")
    else:
        st.warning("Please enter a prompt!")

st.markdown("---")
st.caption("Made with 💫 using Stable Diffusion + Streamlit")
