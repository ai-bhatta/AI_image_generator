import streamlit as st
from PIL import Image
from io import BytesIO

# ---------------- Stable Diffusion imports ----------------
try:
    from diffusers import StableDiffusionPipeline
    import torch
except:
    pass  # If GPU model is not installed, Stable Diffusion section will be disabled

# --- PAGE CONFIG ---
st.set_page_config(page_title="🎨 AI Image Generator", page_icon="✨", layout="centered")
st.title("🎨 AI Image Generator")
st.write("Generate images from text using **Stable Diffusion GPU** or **CPU-friendly mini version**")

st.markdown("---")

# ---------------- Stable Diffusion GPU Section ----------------
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
    st.warning("Stable Diffusion GPU not available. Make sure 'diffusers' and 'torch' are installed and GPU is enabled.")
    st.info("You can still use the CPU-friendly mini version below.")

st.markdown("---")

# ---------------- CPU-Friendly Mini Section ----------------
st.header("🐱 CPU-Friendly Stable Diffusion Mini (3 Images per Prompt)")

cpu_prompt = st.text_input("Enter prompt for CPU-friendly generator:", "A cat making pizza", key="cpu")

if st.button("Generate 3 Images (CPU Mini)"):
    if cpu_prompt.strip():
        with st.spinner("Generating CPU-friendly images..."):
            try:
                from diffusers import StableDiffusionPipeline

                # Use a smaller CPU-friendly model
                model_id = "CompVis/stable-diffusion-v1-4"
                pipe_cpu = StableDiffusionPipeline.from_pretrained(model_id)
                pipe_cpu = pipe_cpu.to("cpu")

                images = [pipe_cpu(cpu_prompt).images[0] for _ in range(3)]
                st.image(images, caption=["Result 1", "Result 2", "Result 3"], use_container_width=True)

                for idx, img in enumerate(images):
                    img.save(f"cpu_image_{idx+1}.png")
                st.success("✅ CPU-friendly images generated and saved!")
            except Exception as e:
                st.error(f"Error generating CPU images: {e}")
    else:
        st.warning("Please enter a prompt for CPU-friendly generation.")

st.markdown("---")
st.caption("Made with 💫 using Stable Diffusion + Streamlit (GPU & CPU)")
