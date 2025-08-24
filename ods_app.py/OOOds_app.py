import os
import zlib
import uuid
import json
import typing as T
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from io import BytesIO

try:
    import magic
except Exception:
    magic = None

APP_VERSION = "1.1.0"
STORE_DIR = "ods_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.json")
FASTA_WRAP = 80
ENCODING_SCHEME = {0b00: "A", 0b01: "C", 0b10: "G", 0b11: "T"}
DECODING_SCHEME = {v: k for k, v in ENCODING_SCHEME.items()}
MAGIC = b"ODS1\x00"

# ===============================================
# Store Management
# ===============================================
def ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    if not os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump({"projects": []}, f, indent=2)

def load_index() -> dict:
    ensure_store()
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_index(idx: dict) -> None:
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

# ===============================================
# Header Definition
# ===============================================
@dataclass
class ODSHeader:
    version: str
    project_id: str
    filename: str
    mimetype: str
    length: int
    compressed: bool
    crc32: int

    def to_bytes(self) -> bytes:
        payload = json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")
        size = len(payload).to_bytes(4, "big")
        return MAGIC + size + payload

    @staticmethod
    def from_bytes(buf: bytes) -> T.Tuple["ODSHeader", int]:
        if not buf.startswith(MAGIC):
            raise ValueError("Not an ODS payload (magic mismatch)")
        size = int.from_bytes(buf[5:9], "big")
        js = json.loads(buf[9:9+size].decode("utf-8"))
        hdr = ODSHeader(**js)
        return hdr, 9 + size

# ===============================================
# DNA Conversion
# ===============================================
def bytes_to_dna(data: bytes) -> str:
    dna_chars = []
    for b in data:
        for shift in (6, 4, 2, 0):
            dna_chars.append(ENCODING_SCHEME[(b >> shift) & 0b11])
    return "".join(dna_chars)

def dna_to_bytes(seq: str) -> bytes:
    if len(seq) % 4 != 0:
        raise ValueError("DNA length must be a multiple of 4 for byte-aligned decoding")
    allowed = set("ACGT")
    if any(ch not in allowed for ch in seq):
        raise ValueError("DNA contains invalid symbols; only A,C,G,T are allowed")
    out = bytearray()
    for i in range(0, len(seq), 4):
        b = 0
        quad = seq[i:i+4]
        for j, base in enumerate(quad):
            b |= (DECODING_SCHEME[base] & 0b11) << (6 - 2*j)
        out.append(b)
    return bytes(out)

# ===============================================
# Analysis Helpers
# ===============================================
def sliding_windows(seq: str, window: int, step: int) -> T.Iterable[str]:
    if window > len(seq) and len(seq) > 0:
        yield seq
        return
    for i in range(0, max(0, len(seq) - window + 1), step):
        yield seq[i:i+window]

def index_of_coincidence(window_seq: str, alphabet: T.Sequence[str] = ("A","C","G","T")) -> float:
    N = len(window_seq)
    if N <= 1:
        return 0.0
    counts = {a: 0 for a in alphabet}
    for ch in window_seq:
        counts[ch] = counts.get(ch, 0) + 1
    num = sum(n*(n-1) for n in counts.values())
    den = N*(N-1)
    return num/den

def cg_content(window_seq: str) -> float:
    if not window_seq:
        return 0.0
    return (window_seq.count("C") + window_seq.count("G")) / len(window_seq)

def make_stain_plots(seq: str, window: int = 300, step: int = 30, project_dir: str = ".") -> dict:
    os.makedirs(project_dir, exist_ok=True)
    ic_values = []
    cg_values = []
    for w in sliding_windows(seq, window, step):
        ic_values.append(index_of_coincidence(w))
        cg_values.append(cg_content(w))
    if len(ic_values) == 0:
        ic_values = [0.0]
        cg_values = [0.0]
    x = np.arange(len(ic_values))
    fig1 = plt.figure(figsize=(9, 3))
    plt.plot(x, ic_values, label="IC")
    plt.plot(x, cg_values, label="CG fraction")
    plt.title("ODS – IC & CG% over sliding windows")
    plt.xlabel(f"Window index (window={window}, step={step})")
    plt.ylabel("Value")
    plt.legend()
    path1 = os.path.join(project_dir, "trace_ic_cg.png")
    fig1.tight_layout()
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    fig2 = plt.figure(figsize=(4, 4))
    plt.scatter(cg_values, ic_values, s=6)
    plt.title("ODS – IC vs CG")
    plt.xlabel("CG fraction")
    plt.ylabel("Index of Coincidence")
    path2 = os.path.join(project_dir, "scatter_ic_vs_cg.png")
    fig2.tight_layout()
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    counts = np.zeros((4,4), dtype=float)
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        if a in "ACGT" and b in "ACGT":
            counts["ACGT".index(a), "ACGT".index(b)] += 1
    if counts.sum() > 0:
        counts = counts / counts.sum()
    fig3 = plt.figure(figsize=(4.5, 4))
    plt.imshow(counts, aspect='equal')
    plt.xticks(range(4), list("ACGT"))
    plt.yticks(range(4), list("ACGT"))
    plt.title("ODS – Dinucleotide heatmap")
    plt.colorbar(label="Probability")
    path3 = os.path.join(project_dir, "heatmap_dinuc.png")
    fig3.tight_layout()
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    return {"trace": path1, "scatter": path2, "heatmap": path3}

# ===============================================
# MIME Detection
# ===============================================
def detect_mime(name: str, data: bytes) -> str:
    if magic is not None:
        try:
            m = magic.Magic(mime=True)
            return m.from_buffer(data)
        except Exception:
            pass
    ext = os.path.splitext(name)[1].lower()
    if ext in {".txt", ".md", ".csv"}: return "text/plain"
    if ext in {".png"}: return "image/png"
    if ext in {".jpg", ".jpeg"}: return "image/jpeg"
    if ext in {".gif"}: return "image/gif"
    if ext in {".mp3"}: return "audio/mpeg"
    if ext in {".wav"}: return "audio/wav"
    if ext in {".mp4"}: return "video/mp4"
    return "application/octet-stream"

# ===============================================
# Compression Utilities
# ===============================================
def compress_image_keep_format(name: str, data: bytes, quality: int = 85) -> bytes:
    """
    Compress image without changing its format or dimensions.
    """
    try:
        img = Image.open(BytesIO(data))
        img_format = img.format
        buf = BytesIO()
        if img_format == "PNG":
            img.save(buf, format=img_format, optimize=True)
        else:
            img.save(buf, format=img_format, optimize=True, quality=quality)
        return buf.getvalue()
    except Exception:
        # Not an image or cannot compress; return original
        return data

def estimate_compression(name: str, data: bytes, quality: int = 85) -> T.Tuple[bool, int, int, float]:
    """
    Estimate compression savings for image files.
    Returns: (will_compress, original_size, estimated_size, reduction%)
    """
    try:
        compressed = compress_image_keep_format(name, data, quality)
        original_size = len(data)
        compressed_size = len(compressed)
        reduction = round(((original_size - compressed_size) / original_size) * 100, 2)
        return compressed_size < original_size, original_size, compressed_size, reduction
    except Exception:
        return False, len(data), len(data), 0.0

# ===============================================
# Pack & Unpack
# ===============================================
def pack_to_dna(name: str, data: bytes, compress: bool, quality: int = 85) -> T.Tuple[ODSHeader, str]:
    mimetype = detect_mime(name, data)

    if compress and mimetype.startswith("image/"):
        data = compress_image_keep_format(name, data, quality)

    crc = zlib.crc32(data) & 0xFFFFFFFF
    hdr = ODSHeader(
        version=APP_VERSION,
        project_id=str(uuid.uuid4()),
        filename=name,
        mimetype=mimetype,
        length=len(data),
        compressed=compress,
        crc32=crc,
    )
    packet = hdr.to_bytes() + data
    dna = bytes_to_dna(packet)
    return hdr, dna

def unpack_from_dna(dna: str) -> T.Tuple[ODSHeader, bytes]:
    packet = dna_to_bytes(dna)
    hdr, ofs = ODSHeader.from_bytes(packet)
    payload = packet[ofs:ofs + hdr.length]
    if len(payload) != hdr.length:
        raise ValueError("Payload truncated")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != hdr.crc32:
        raise ValueError("CRC mismatch – data corrupted or wrong sequence")
    return hdr, payload

def to_fasta(seq_id: str, dna: str, wrap: int = FASTA_WRAP) -> str:
    lines = [f">{seq_id}"]
    for i in range(0, len(dna), wrap):
        lines.append(dna[i:i+wrap])
    return "\n".join(lines) + "\n"

# ===============================================
# Streamlit UI
# ===============================================
st.set_page_config(page_title="Objective Digital Stains (ODS)", layout="wide")
st.title("🧬 Objective Digital Stains (ODS)")
st.caption("Convert any file into a DNA-like sequence, generate stain plots, and decode it back later.")

with st.sidebar:
    st.header("1) Encode")
    uploaded = st.file_uploader("Upload any file", type=None)
    compress = st.checkbox("Compress image (keep format & size)", value=True)
    quality = st.slider("Compression quality (for JPG)", min_value=10, max_value=95, value=85)
    window = st.number_input("Sliding window length", value=300, min_value=50, step=10)
    step = st.number_input("Window step", value=30, min_value=1, step=1)
    check_compression = st.button("Check compression potential")
    do_encode = st.button("Encode to DNA + Generate Stains")
    st.header("2) Decode")
    seq_input = st.text_area("Paste DNA sequence (FASTA or raw)", height=140)
    do_decode = st.button("Decode DNA back to file")

ensure_store()
col1, col2 = st.columns(2)

if uploaded is not None and check_compression:
    raw = uploaded.read()
    will_compress, orig, comp, reduction = estimate_compression(uploaded.name, raw, quality)
    if will_compress:
        st.success(f"Compression possible: {orig//1024}KB → {comp//1024}KB ({reduction}% reduction)")
    else:
        st.info(f"No significant compression benefit. Original size: {orig//1024}KB")

if do_encode and uploaded is not None:
    name = uploaded.name
    raw = uploaded.read()
    if raw is None:
        st.error("Empty file")
    else:
        hdr, dna = pack_to_dna(name, raw, compress, quality)
        project_dir = os.path.join(STORE_DIR, hdr.project_id)
        os.makedirs(project_dir, exist_ok=True)
        fasta = to_fasta(hdr.project_id, dna)
        fasta_path = os.path.join(project_dir, f"{hdr.project_id}.fasta")
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(fasta)
        meta = asdict(hdr)
        meta.update({"dna_length": len(dna)})
        with open(os.path.join(project_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        paths = make_stain_plots(dna, window=int(window), step=int(step), project_dir=project_dir)
        idx = load_index()
        idx.setdefault("projects", []).append(meta)
        save_index(idx)
        with col1:
            st.subheader("DNA Output")
            st.text_area("FASTA", fasta, height=220)
            st.download_button("Download FASTA", data=fasta, file_name=f"{hdr.project_id}.fasta", mime="text/plain")
            st.code(json.dumps(meta, indent=2), language="json")
        with col2:
            st.subheader("Objective Digital Stains")
            st.image(paths["trace"], caption="IC & CG% traces", use_column_width=True)
            st.image(paths["scatter"], caption="IC vs CG scatter", use_column_width=True)
            st.image(paths["heatmap"], caption="Dinucleotide heatmap", use_column_width=True)

if do_decode and seq_input.strip():
    raw_seq = []
    for line in seq_input.splitlines():
        line = line.strip()
        if not line or line.startswith('>'):
            continue
        raw_seq.append(line)
    dna_seq = "".join(raw_seq).upper().replace(" ", "")
    try:
        hdr, data = unpack_from_dna(dna_seq)
        st.success("Decoded successfully!")
        st.json(asdict(hdr))
        st.download_button(label=f"Download decoded: {hdr.filename}", data=data, file_name=hdr.filename, mime=hdr.mimetype or "application/octet-stream")
    except Exception as e:
        st.error(f"Decoding failed: {e}")

st.markdown("---")
st.subheader("📁 Stored Projects")

idx = load_index()
if not idx.get("projects"):
    st.info("No projects yet – encode a file to get started.")
else:
    df = pd.DataFrame(idx["projects"]).sort_values("project_id", ascending=False)
    st.dataframe(df, use_container_width=True)
    sel = st.selectbox("Open project", ["(choose)"] + list(df["project_id"]))
    if sel != "(choose)":
        pdir = os.path.join(STORE_DIR, sel)
        try:
            st.write(f"**Directory**: {pdir}")
            with open(os.path.join(pdir, "meta.json"), "r", encoding="utf-8") as f:
                st.code(f.read(), language="json")
            for img in ["trace_ic_cg.png", "scatter_ic_vs_cg.png", "heatmap_dinuc.png"]:
                path = os.path.join(pdir, img)
                if os.path.exists(path):
                    st.image(path, caption=img, use_column_width=True)
            fasta_path = os.path.join(pdir, f"{sel}.fasta")
            if os.path.exists(fasta_path):
                with open(fasta_path, "r", encoding="utf-8") as f:
                    fasta_txt = f.read()
                st.download_button("Download FASTA again", data=fasta_txt, file_name=f"{sel}.fasta")
        except Exception as ex:
            st.error(f"Failed to open project: {ex}")

st.markdown(f"""
---
**ODS v{APP_VERSION}** · Deterministic bytes↔DNA mapping (2-bit per base) with metadata & CRC.
Enhanced with image compression (lossless for PNG, quality-based for JPG) and compression check.
""")
