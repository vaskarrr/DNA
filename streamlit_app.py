import streamlit as st
import subprocess
import os
from PIL import Image
import numpy as np
from picdna_sim.primers import PrimerScheme
from picdna_sim.core import progressive_layers, make_oligos_for_layer, mse

st.set_page_config("PIC-DNA++ Hub", layout="wide")

st.title("PIC-DNA++ : Progressive + Random-Access DNA Image Storage")

tab1, tab2 = st.tabs(["📦 Batch Simulation", "🔍 Interactive Random Access"])

# ------------------------
# TAB 1: Batch Simulation
# ------------------------
with tab1:
    st.subheader("Run Batch Simulation (CLI in the background)")

    img_folder = st.text_input("Image folder path", "./sample_images")
    out_folder = st.text_input("Output folder path", "./out")
    layers = st.slider("Number of progressive layers", 3, 6, 5)

    if st.button("Run Batch Simulation"):
        cmd = f"python -m picdna_sim.simulate --images \"{img_folder}\" --out \"{out_folder}\" --layers {layers}"
        with st.spinner(f"Running simulation for {img_folder} ..."):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Simulation completed successfully!")
            report_file = os.path.join(out_folder, "report.json")
            if os.path.exists(report_file):
                st.subheader("Simulation Report")
                with open(report_file) as f:
                    st.json(f.read())
        else:
            st.error("Simulation failed!")
            st.code(result.stderr)

# ------------------------
# TAB 2: Interactive Random Access
# ------------------------
with tab2:
    st.subheader("Upload Images for Progressive & Random Access Demo")

    images = st.file_uploader("Upload one or more images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    n_layers = st.slider("Number of progressive layers", 3, 6, 5, key="interactive_layers")

    if images:
        pil_images = [Image.open(f).convert("RGB") for f in images]
        primers = PrimerScheme(n_images=len(pil_images), n_layers=n_layers)

        st.subheader("Thumbnails (L0) for Random Access")
        thumbs = []
        for i, im in enumerate(pil_images):
            layers_list = progressive_layers(im, n_layers=n_layers)
            thumbs.append(layers_list[0])

        cols = st.columns(len(thumbs))
        for c, th in zip(cols, thumbs):
            c.image(th, caption=f"Image {thumbs.index(th)} L0")

        img_idx = st.number_input("Select image id", 0, len(thumbs)-1, 0)
        target_layer = st.slider("Target layer (higher = better quality)", 0, n_layers-1, n_layers-1)

        if st.button("Simulate Random-Access Decode"):
            im = pil_images[img_idx]
            layers_list = progressive_layers(im, n_layers=n_layers)
            pools = []
            for k, layer_img in enumerate(layers_list):
                oligos = make_oligos_for_layer(img_idx, k, np.array(layer_img), primers, data_block_nt=200)
                pools.append(oligos)

            total_oligos_thumbs = sum(len(pools[0]) for _ in pil_images)
            total_oligos_selected = sum(len(pools[k]) for k in range(1, target_layer+1))
            st.metric("Approx. read cost (oligos)", total_oligos_thumbs + total_oligos_selected)

            baseline = np.array(layers_list[-1])
            up = layers_list[target_layer].resize(im.size, Image.BICUBIC)
            dist = mse(np.array(up), baseline)
            st.metric("Distortion proxy (MSE, lower better)", f"{dist:.1f}")

            st.image([layers_list[0], layers_list[target_layer], layers_list[-1]], caption=["L0", f"L{target_layer}", "Full (Lmax)"])
