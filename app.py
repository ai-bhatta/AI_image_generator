import streamlit as st
from PIL import Image
from io import BytesIO
import requests

# --- Stable Diffusion imports ---
try:
    from diffusers import StableDiffusionPipeline
    import torch
except:
    pass  # If GPU model is not installed, Stable Diffusion section will be disabled

# --- PAGE CONFIG ---
st.set_page_config(page_title="🎨 AI Image Generator", page_icon="✨", layout="centered")
st.title("🎨 AI Image Generator")
st.write("Generate images from text using **Stable Diffusion** (GPU) or **Craiyon** (CPU, fast)")

st.markdown("---")

# ---------------- Stable Diffusion Section ----------------
st.header("💎 Stable Diffusion (High Quality, GPU Required)")

try:
    @st.cache_resource
    def load_sd_model():
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe

    pipe = load_sd_model()

    sd_prompt = st.text_input("Enter prompt for Stable Diffusion:", "A futuristic city floating in the clouds", key="sd")
    if st.button("Generate Image (Stable Diffusion)"):
        if sd_prompt.strip():
            with st.spinner("Generating image with Stable Diffusion..."):
                image = pipe(sd_prompt).images[0]
                st.image(image, caption=f"Generated: {sd_prompt}", use_container_width=True)
                image.save("stable_diffusion_image.png")
                st.success("✅ Image generated and saved as stable_diffusion_image.png")
        else:
            st.warning("Please enter a prompt for Stable Diffusion.")
except Exception as e:
    st.warning("Stable Diffusion not available. Make sure 'diffusers' and 'torch' are installed and GPU is enabled.")
    st.info("You can still use Craiyon (CPU-friendly) below.")

st.markdown("---")

# ---------------- Craiyon Section ----------------
st.header("🐱 Craiyon / DALL·E Mini (CPU, Fast)")

craiyon_prompt = st.text_input("Enter prompt for Craiyon:", "A cat riding a skateboard", key="craiyon")
if st.button("Generate Images (Craiyon)"):
    if craiyon_prompt.strip():
        with st.spinner("Generating images with Craiyon..."):
            try:
                res = requests.post("https://api.craiyon.com/v1", json={"prompt": craiyon_prompt})
                images = res.json()["images"]
                pil_images = [Image.open(BytesIO(bytes.fromhex(i))) for i in images[:3]]
                st.image(pil_images, caption=["Result 1", "Result 2", "Result 3"], use_container_width=True)
                st.success("✅ Craiyon images generated!")
            except Exception as e:
                st.error(f"Error generating Craiyon images: {e}")
    else:
        st.warning("Please enter a prompt for Craiyon.")

st.markdown("---")
st.caption("Made with 💫 using Stable Diffusion + Craiyon + Streamlit")
