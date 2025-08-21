import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
import string

st.set_page_config(page_title="DNA Storage App", layout="wide")

st.title("DNA Data Storage: Motifs, ECC, and Plasmid Mapping")

uploaded_file = st.file_uploader("Upload a DNA sequence (FASTA or TXT)", type=["txt", "fasta"])

if uploaded_file:
    seq = uploaded_file.read().decode().upper()
    seq = "".join([c for c in seq if c in "ACGT"])
    st.subheader("Uploaded DNA Sequence")
    st.write(seq[:200] + ("..." if len(seq) > 200 else ""))

    st.subheader("Motif Analysis")
    motif = st.text_input("Enter motif to search (e.g., ATG)", "ATG")
    if motif:
        positions = [i for i in range(len(seq)) if seq.startswith(motif, i)]
        st.write(f"Motif '{motif}' found {len(positions)} times")
        fig, ax = plt.subplots()
        ax.plot(range(len(seq)), [0]*len(seq), "-", alpha=0)
        ax.scatter(positions, [0]*len(positions), color="red", label=f"{motif} positions")
        ax.set_title("Motif Positions")
        ax.legend(loc="upper right")
        st.pyplot(fig)

    st.subheader("Error-Correcting Codes (ECC)")
    def add_parity_bits(binary_str):
        return binary_str + str(binary_str.count("1") % 2)

    text = st.text_input("Enter text to encode in DNA", "HELLO")
    if text:
        binary = "".join(format(ord(c), "08b") for c in text)
        with_parity = add_parity_bits(binary)
        mapping = {"00":"A", "01":"C", "10":"G", "11":"T"}
        dna_encoded = "".join(mapping[binary[i:i+2]] for i in range(0, len(with_parity), 2) if len(binary[i:i+2])==2)
        st.write("Binary:", binary)
        st.write("With parity:", with_parity)
        st.write("DNA encoded:", dna_encoded)

    st.subheader("Plasmid Mapping")
    fig = go.Figure(data=[go.Scatterpolar(r=[1]*len(seq), theta=list(np.linspace(0, 360, len(seq))), mode='markers', marker=dict(size=2))])
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("K-mer Analysis")
    def kmer_count(seq, k):
        counts = {}
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            counts[kmer] = counts.get(kmer, 0)+1
        return counts

    k2 = kmer_count(seq,2)
    df2 = pd.DataFrame([[a[0], a[1], c] for a,c in k2.items()], columns=["Base1","Base2","Count"])
    heatmap = df2.pivot(index="Base1", columns="Base2", values="Count")
    fig, ax = plt.subplots()
    cax = ax.matshow(heatmap.fillna(0))
    fig.colorbar(cax)
    ax.set_title("2-mer Heatmap")
    ax.legend(["Frequency"], loc="upper right")
    st.pyplot(fig)

    k3 = kmer_count(seq,3)
    top3 = sorted(k3.items(), key=lambda x: -x[1])[:10]
    df3 = pd.DataFrame(top3, columns=["3-mer","Count"])
    fig, ax = plt.subplots()
    ax.bar(df3["3-mer"], df3["Count"])
    ax.set_title("Top 10 3-mers")
    ax.legend(["Count"], loc="upper right")
    st.pyplot(fig)

else:
    st.info("Upload a DNA sequence file to begin analysis.")
