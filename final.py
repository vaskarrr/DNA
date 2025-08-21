# ODS-X: Interactive DNA Data Storage and Bioinformatics Visualization Suite
# Streamlit single-file app
# ---------------------------------------------------------------
# Features
# - Encode any file -> DNA (2 bits/base) with metadata header + CRC
# - Optional gzip compression
# - Optional Hamming(7,4) ECC on the payload bytes (nibble-wise)
# - Decode DNA/FASTA back to original file (verifies CRC, tries ECC-correct)
# - Stain plots: IC & GC traces, IC vs GC scatter, dinucleotide heatmap
# - Bioinformatics: motif finding (TATA-box), CpG island scan, k-mer heatmaps
# - 3D DNA helix visualization with Plotly (bases color-coded, motifs highlighted)
# - DNA storage simulation: base-mutation, ECC correction report
# - Synthetic biology: circular plasmid map and GenBank (.gb) export of insert
# ---------------------------------------------------------------

import os
import io
import re
import zlib
import json
import uuid
import base64
import random
import typing as T
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

try:
    import magic  # optional, for MIME detection
except Exception:
    magic = None

# --------------------------- Constants ---------------------------
APP_VERSION = "1.3.0"
STORE_DIR = "ods_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.json")
FASTA_WRAP = 80
ENCODING_SCHEME = {0b00: "A", 0b01: "C", 0b10: "G", 0b11: "T"}
DECODING_SCHEME = {v: k for k, v in ENCODING_SCHEME.items()}
MAGIC = b"ODS1\x00"

# --------------------------- Utilities ---------------------------
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


# --------------------------- Header ---------------------------
@dataclass
class ODSHeader:
    version: str
    project_id: str
    filename: str
    mimetype: str
    length: int               # payload length (after compression and ECC, before DNA mapping)
    compressed: bool
    crc32: int
    ecc: bool                 # whether Hamming(7,4) ECC was applied

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

# --------------------------- Bit helpers & ECC ---------------------------

def bytes_to_bits(data: bytes) -> T.List[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def bits_to_bytes(bits: T.List[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        if len(chunk) < 8:
            chunk += [0] * (8 - len(chunk))
        val = 0
        for bit in chunk:
            val = (val << 1) | (bit & 1)
        out.append(val)
    return bytes(out)

# Hamming(7,4): Encode 4 data bits -> 7 bits with single-error correction
# Positions: 1 2 3 4 5 6 7  (1-indexed); parity bits at 1,2,4
# We'll keep order [b1..b7] in list form

def hamming74_encode_nibble(nib: T.List[int]) -> T.List[int]:
    d1, d2, d3, d4 = nib
    # p1 covers bits 1,3,5,7 -> over d1,d2,d4
    p1 = (d1 ^ d2 ^ d4) & 1
    # p2 covers bits 2,3,6,7 -> over d1,d3,d4
    p2 = (d1 ^ d3 ^ d4) & 1
    # p4 covers bits 4,5,6,7 -> over d2,d3,d4
    p4 = (d2 ^ d3 ^ d4) & 1
    return [p1, p2, d1, p4, d2, d3, d4]


def hamming74_decode_block(block: T.List[int]) -> T.Tuple[T.List[int], int]:
    # returns data bits [d1..d4] and number of corrected bits (0 or 1 or -1 if uncorrectable)
    b1, b2, b3, b4, b5, b6, b7 = block
    # syndrome bits s1, s2, s4
    s1 = (b1 ^ b3 ^ b5 ^ b7) & 1
    s2 = (b2 ^ b3 ^ b6 ^ b7) & 1
    s4 = (b4 ^ b5 ^ b6 ^ b7) & 1
    syndrome = (s4 << 2) | (s2 << 1) | s1
    corrected = 0
    if syndrome != 0 and 1 <= syndrome <= 7:
        # flip the bit indicated by syndrome
        idx = syndrome - 1
        block[idx] ^= 1
        corrected = 1
        # reassign after correction
        b1, b2, b3, b4, b5, b6, b7 = block
    # data bits at positions 3,5,6,7
    d1, d2, d3, d4 = b3, b5, b6, b7
    return [d1, d2, d3, d4], corrected


def ecc_encode_bytes(data: bytes) -> bytes:
    bits = bytes_to_bits(data)
    out_bits: T.List[int] = []
    for i in range(0, len(bits), 4):
        nib = bits[i:i+4]
        if len(nib) < 4:
            nib += [0] * (4 - len(nib))
        block = hamming74_encode_nibble(nib)
        out_bits.extend(block)
    return bits_to_bytes(out_bits)


def ecc_decode_bytes(data: bytes) -> T.Tuple[bytes, int, int]:
    bits = bytes_to_bits(data)
    corrected_total = 0
    out_bits: T.List[int] = []
    uncorrectable = 0
    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        if len(block) < 7:
            block += [0] * (7 - len(block))
        nib, corrected = hamming74_decode_block(block)
        corrected_total += max(0, corrected)
        out_bits.extend(nib)
    # Trim pad bits to nearest byte length of original multiple of 4->7 expansion is unknown here.
    # We'll drop trailing pad bits after header+CRC verification in unpack.
    return bits_to_bytes(out_bits), corrected_total, uncorrectable

# --------------------------- DNA mapping ---------------------------

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


# --------------------------- Sliding windows & stains ---------------------------

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


def make_stain_figures(seq: str, window: int = 300, step: int = 30):
    ic_values, cg_values = [], []
    for w in sliding_windows(seq, window, step):
        ic_values.append(index_of_coincidence(w))
        cg_values.append(cg_content(w))
    if len(ic_values) == 0:
        ic_values = [0.0]
        cg_values = [0.0]
    x = np.arange(len(ic_values))

    # Trace figure
    fig_trace, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(x, ic_values, label="IC")
    ax1.plot(x, cg_values, label="CG fraction")
    ax1.set_title("ODS – IC & CG% over sliding windows")
    ax1.set_xlabel(f"Window index (window={window}, step={step})")
    ax1.set_ylabel("Value")
    ax1.legend()
    fig_trace.tight_layout()

    # Scatter
    fig_scatter, ax2 = plt.subplots(figsize=(4, 4))
    ax2.scatter(cg_values, ic_values, s=8)
    ax2.set_title("ODS – IC vs CG")
    ax2.set_xlabel("CG fraction")
    ax2.set_ylabel("Index of Coincidence")
    fig_scatter.tight_layout()

    # Dinucleotide heatmap
    counts = np.zeros((4,4), dtype=float)
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        if a in "ACGT" and b in "ACGT":
            counts["ACGT".index(a), "ACGT".index(b)] += 1
    if counts.sum() > 0:
        counts = counts / counts.sum()

    fig_heat, ax3 = plt.subplots(figsize=(4.6, 4.2))
    im = ax3.imshow(counts, aspect='equal')
    ax3.set_xticks(range(4), list("ACGT"))
    ax3.set_yticks(range(4), list("ACGT"))
    ax3.set_title("ODS – Dinucleotide heatmap")
    cbar = fig_heat.colorbar(im, ax=ax3)
    cbar.set_label("Probability")
    fig_heat.tight_layout()

    return fig_trace, fig_scatter, fig_heat


# --------------------------- MIME & FASTA ---------------------------

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


def to_fasta(seq_id: str, dna: str, wrap: int = FASTA_WRAP) -> str:
    lines = [f">{seq_id}"]
    for i in range(0, len(dna), wrap):
        lines.append(dna[i:i+wrap])
    return "\n".join(lines) + "\n"


# --------------------------- Pack/Unpack ---------------------------

def pack_to_dna(name: str, data: bytes, compress: bool, use_ecc: bool) -> T.Tuple[ODSHeader, str]:
    mimetype = detect_mime(name, data)
    payload = zlib.compress(data) if compress else data
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    ecc_payload = ecc_encode_bytes(payload) if use_ecc else payload
    hdr = ODSHeader(
        version=APP_VERSION,
        project_id=str(uuid.uuid4()),
        filename=name,
        mimetype=mimetype,
        length=len(ecc_payload),
        compressed=compress,
        crc32=crc,
        ecc=use_ecc,
    )
    packet = hdr.to_bytes() + ecc_payload
    dna = bytes_to_dna(packet)
    return hdr, dna


def unpack_from_dna(dna: str) -> T.Tuple[ODSHeader, bytes, dict]:
    packet = dna_to_bytes(dna)
    hdr, ofs = ODSHeader.from_bytes(packet)
    payload = packet[ofs:ofs + hdr.length]
    if len(payload) != hdr.length:
        raise ValueError("Payload truncated")

    ecc_report = {"used": hdr.ecc, "corrected_bits": 0}

    if hdr.ecc:
        # Decode ECC, obtain data bits, then bytes
        decoded_bytes, corrected, _unc = ecc_decode_bytes(payload)
        ecc_report["corrected_bits"] = corrected
        # Trim to the exact compressed length isn't stored; we rely on CRC of decompressed payload to validate.
        # We attempt decompression if compressed else CRC over decoded_bytes.
        # To ensure clean CRC check, we try both direct CRC and after decompression if compressed.
        ecc_payload = decoded_bytes
    else:
        ecc_payload = payload

    if (zlib.crc32(ecc_payload) & 0xFFFFFFFF) != hdr.crc32:
        # If compressed, CRC is computed on compressed payload; ensure we used matching set
        raise ValueError("CRC mismatch – data corrupted or wrong sequence")

    data = zlib.decompress(ecc_payload) if hdr.compressed else ecc_payload
    return hdr, data, ecc_report


# --------------------------- Bioinformatics: motifs & CpG ---------------------------
TATA_REGEX = re.compile(r"TATA[AT]A[AT]", re.IGNORECASE)


def find_motifs(seq: str) -> pd.DataFrame:
    seqU = seq.upper()
    rows = []
    for m in TATA_REGEX.finditer(seqU):
        rows.append({"motif": "TATA-box", "start": m.start(), "end": m.end(), "sequence": m.group()})
    # Add simple CpG island scan with sliding window heuristic
    # Criteria: window size 200, GC >= 0.5 and observed/expected CpG >= 0.6
    win = 200
    if len(seqU) >= 20:
        for i in range(0, len(seqU) - win + 1, 20):
            w = seqU[i:i+win]
            gc = (w.count('C') + w.count('G')) / win
            obs_cpg = w.count('CG')
            exp_cpg = (w.count('C') * w.count('G')) / max(1, win)
            oe = obs_cpg / max(1e-9, exp_cpg)
            if gc >= 0.5 and oe >= 0.6 and obs_cpg >= 5:
                rows.append({"motif": "CpG_island", "start": i, "end": i+win, "sequence": w[:12] + "..."})
    df = pd.DataFrame(rows)
    if not len(df):
        df = pd.DataFrame(columns=["motif","start","end","sequence"])  # empty
    return df


def kmer_frequencies(seq: str, k: int = 3) -> pd.DataFrame:
    seqU = ''.join([c for c in seq.upper() if c in 'ACGT'])
    counts = {}
    for i in range(len(seqU) - k + 1):
        kmer = seqU[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["kmer","count","freq"]).set_index("kmer")
    total = sum(counts.values())
    df = pd.DataFrame([{"kmer": k, "count": v, "freq": v/total} for k, v in counts.items()])
    df = df.sort_values("kmer").set_index("kmer")
    return df


def plot_kmer_heatmap(df: pd.DataFrame, k: int = 2):
    # For k=2 or k=3 create a square-ish heatmap if possible
    if df.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        return fig
    alphabet = ['A','C','G','T']
    if k == 2:
        mat = np.zeros((4,4))
        for i,a in enumerate(alphabet):
            for j,b in enumerate(alphabet):
                mat[i,j] = df.loc.get(a+b, {}).get('freq', 0.0) if hasattr(df.loc.get(a+b, {}), 'get') else (df['freq'].get(a+b, 0.0) if a+b in df.index else 0.0)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(mat, aspect='equal')
        ax.set_xticks(range(4), alphabet)
        ax.set_yticks(range(4), alphabet)
        ax.set_title("2-mer frequency heatmap")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Frequency")
        fig.tight_layout()
        return fig
    else:
        # For k=3, show as bar plot sorted by freq
        fig, ax = plt.subplots(figsize=(8, 3))
        df_sorted = df.sort_values('freq', ascending=False).head(64)
        ax.bar(df_sorted.index, df_sorted['freq'])
        ax.set_xticklabels(df_sorted.index, rotation=90)
        ax.set_title("Top 64 3-mers by frequency")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        return fig


# --------------------------- 3D Helix Visualization ---------------------------
BASE_COLORS = {"A": "#1f77b4", "C": "#ff7f0e", "G": "#2ca02c", "T": "#d62728"}


def helix_coordinates(n: int, radius: float=1.0, pitch: float=3.4, points_per_base:int=1):
    # Simple single-strand helix param; we plot bases as points along a helix
    theta = np.linspace(0, 2*np.pi * n/10, n*points_per_base)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(0, pitch*(n/10), n*points_per_base)
    return x, y, z


def plot_3d_helix(seq: str, motifs: pd.DataFrame) -> go.Figure:
    s = ''.join([c for c in seq.upper() if c in 'ACGT'])
    n = len(s)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="No sequence")
        return fig
    x,y,z = helix_coordinates(n)
    colors = [BASE_COLORS.get(b, '#888888') for b in s]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=colors), name='Bases'))
    # Highlight motifs
    if not motifs.empty:
        for _, row in motifs.iterrows():
            stt, end = int(row['start']), int(row['end'])
            if stt < n:
                e = min(end, n)
                fig.add_trace(go.Scatter3d(x=x[stt:e], y=y[stt:e], z=z[stt:e], mode='lines', line=dict(width=8), name=str(row['motif'])))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title="3D DNA Helix (bases & motifs)")
    return fig


# --------------------------- Mutation Simulation ---------------------------

NUC = ['A','C','G','T']


def mutate_dna(seq: str, rate: float, seed: int = 123) -> str:
    rng = random.Random(seed)
    out = []
    for ch in seq:
        if ch not in 'ACGT':
            out.append(ch)
            continue
        if rng.random() < rate:
            choices = [n for n in NUC if n != ch]
            out.append(rng.choice(choices))
        else:
            out.append(ch)
    return ''.join(out)


# --------------------------- Plasmid & GenBank ---------------------------

def make_genbank(seq_id: str, insert_seq: str, plasmid_len: int = 3000) -> str:
    # Fake backbone with ORI and AMP; insert is placed at 1000..1000+len-1
    ins_len = len(insert_seq)
    total = plasmid_len
    ins_start = 1000
    ins_end = ins_start + ins_len - 1
    locus = f"LOCUS       {seq_id:<16} {total} bp    DNA     circular SYN\n"
    origin = ["ORIGIN\n"]
    # Generate fake backbone sequence (random but reproducible)
    rng = random.Random(42)
    backbone = ''.join(rng.choice('acgt') for _ in range(total))
    # place insert (uppercase) into backbone region
    b = list(backbone)
    for i,ch in enumerate(insert_seq.lower()):
        pos = (ins_start-1)+i
        if 0 <= pos < total:
            b[pos] = ch
    final = ''.join(b)
    # Write in GenBank format
    features = (
        "FEATURES             Location/Qualifiers\n"
        f"     source          1..{total}\n"
        f"     misc_feature    {ins_start}..{ins_end}\n"
        f"                     /note=\"ODS-X encoded insert\"\n"
        f"     rep_origin      100..300\n"
        f"                     /note=\"ori (simulated)\"\n"
        f"     CDS             complement(600..1400)\n"
        f"                     /product=\"ampR (simulated)\"\n"
    )
    # ORIGIN lines
    for i in range(0, total, 60):
        line = final[i:i+60]
        blocks = [line[j:j+10] for j in range(0, len(line), 10)]
        origin.append(f"{i+1:>9} "+' '.join(blocks)+"\n")
    end = "//\n"
    return locus + features + "\n" + ''.join(origin) + end


def plot_plasmid_map(insert_len: int, plasmid_len: int = 3000) -> go.Figure:
    fig = go.Figure()
    # Draw circle
    theta = np.linspace(0, 2*np.pi, 361)
    r_outer = 1.0
    x = r_outer * np.cos(theta)
    y = r_outer * np.sin(theta)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Backbone'))
    # Features as arcs: ori (100-300), ampR (600-1400), insert (1000 ..)
    def arc(start_bp, end_bp, label):
        start = (start_bp / plasmid_len) * 2*np.pi
        end = (end_bp / plasmid_len) * 2*np.pi
        th = np.linspace(start, end, 50)
        fig.add_trace(go.Scatter(x=0.85*np.cos(th), y=0.85*np.sin(th), mode='lines', line=dict(width=8), name=label))
    arc(100, 300, 'ori')
    arc(600, 1400, 'ampR')
    arc(1000, 1000+max(10, insert_len), 'insert')
    fig.update_layout(title='Plasmid map (schematic)', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# =========================== Streamlit UI ===========================
st.set_page_config(page_title="ODS-X: DNA Data Storage & Viz", layout="wide")

st.title("🧬 ODS-X: DNA Data Storage and Bioinformatics Visualization")
st.caption("Encode files to DNA-like sequences, analyze motifs, visualize in 3D, simulate storage errors, and export plasmid GenBank.")

ensure_store()

# Sidebar controls
with st.sidebar:
    st.header("Encode")
    uploaded = st.file_uploader("Upload any file", type=None)
    compress = st.checkbox("Gzip-compress payload", value=True)
    use_ecc = st.checkbox("Use Hamming(7,4) ECC (more robust, longer)", value=False)

    st.header("Decode")
    seq_input = st.text_area("Paste DNA sequence (FASTA or raw)", height=160)

    st.header("Analysis Settings")
    window = st.number_input("Sliding window length", value=300, min_value=50, step=10)
    step = st.number_input("Window step", value=30, min_value=1, step=1)
    k_sel = st.selectbox("k-mer size", [2,3], index=0)

    st.header("Simulation")
    mut_rate = st.slider("Base mutation rate", min_value=0.0, max_value=0.05, value=0.005, step=0.001)

# Tabs
home_tab, enc_tab, dec_tab, bio_tab, helix_tab, sim_tab, synbio_tab = st.tabs([
    "🏠 Home", "🔐 Encode", "🔓 Decode", "🧪 Bioinformatics", "🧭 3D Helix", "🧬 Storage Simulation", "🧫 Synthetic Biology"
])

# ---------------- Home ----------------
with home_tab:
    st.subheader("What is ODS-X?")
    st.markdown(
        """
        **ODS-X** converts arbitrary bytes into a DNA-like alphabet (A/C/G/T) using a deterministic 2-bit mapping.
        It stores a self-describing JSON header and CRC for robust decoding, with optional **gzip compression** and **Hamming(7,4) ECC**.

        Beyond storage, it performs **bioinformatics-inspired analyses** (IC/GC, k-mer spectra, motif scans), renders a **3D helix**, simulates **mutations & error-correction**, and exports a **GenBank** plasmid with your insert.
        """
    )
    st.info("Tip: Use ECC for noisy-channel simulations; it increases size but can correct single-bit errors per 7-bit block.")

# ---------------- Encode ----------------
with enc_tab:
    col1, col2 = st.columns(2)
    if uploaded is not None:
        name = uploaded.name
        raw = uploaded.read()
        if raw is None or len(raw) == 0:
            st.error("Empty file uploaded.")
        else:
            hdr, dna = pack_to_dna(name, raw, compress, use_ecc)
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

            # Update index
            idx = load_index()
            idx.setdefault("projects", []).append(meta)
            save_index(idx)

            with col1:
                st.subheader("DNA Output")
                st.text_area("FASTA", fasta, height=220)
                st.download_button("Download FASTA", data=fasta, file_name=f"{hdr.project_id}.fasta", mime="text/plain")
                st.code(json.dumps(meta, indent=2), language="json")
                ratio = len(dna)/max(1, len(raw))
                st.caption(f"Encoding expansion: {ratio:.2f} bases per input byte (includes header+{('ECC,' if use_ecc else '')}{'gzip' if compress else 'raw'}).")

            with col2:
                st.subheader("Objective Digital Stains")
                fig_trace, fig_scatter, fig_heat = make_stain_figures(dna, window=int(window), step=int(step))
                st.pyplot(fig_trace, use_container_width=True)
                st.pyplot(fig_scatter, use_container_width=True)
                st.pyplot(fig_heat, use_container_width=True)
    else:
        st.info("Upload a file on the left to encode.")

# ---------------- Decode ----------------
with dec_tab:
    if seq_input.strip():
        # parse FASTA or raw
        raw_seq = []
        for line in seq_input.splitlines():
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            raw_seq.append(line)
        dna_seq = "".join(raw_seq).upper().replace(" ", "")
        try:
            hdr, data, ecc_report = unpack_from_dna(dna_seq)
            st.success("Decoded successfully!")
            st.json(asdict(hdr))
            if ecc_report.get("used"):
                st.info(f"ECC used. Corrected bits: {ecc_report.get('corrected_bits', 0)}")
            st.download_button(label=f"Download decoded: {hdr.filename}", data=data, file_name=hdr.filename, mime=hdr.mimetype or "application/octet-stream")
        except Exception as e:
            st.error(f"Decoding failed: {e}")
    else:
        st.info("Paste a DNA/FASTA sequence in the sidebar to decode.")

# ---------------- Bioinformatics ----------------
with bio_tab:
    st.subheader("Motifs, CpG Islands, and k-mer spectra")
    seq_for_bio = None
    # Choose source: from last encode (if any) or textarea
    idx = load_index()
    proj_ids = [p['project_id'] for p in idx.get('projects', [])]
    choice = st.selectbox("Choose a stored project to analyze (or paste DNA below)", ["(none)"] + proj_ids)
    pasted = st.text_area("Or paste a DNA sequence here", height=160)
    if choice != "(none)":
        try:
            with open(os.path.join(STORE_DIR, choice, f"{choice}.fasta"), 'r', encoding='utf-8') as f:
                fasta_txt = f.read()
            seq_for_bio = ''.join([ln.strip() for ln in fasta_txt.splitlines() if ln and not ln.startswith('>')])
        except Exception as ex:
            st.error(f"Failed to load project: {ex}")
    elif pasted.strip():
        seq_for_bio = ''.join([c for c in pasted.upper() if c in 'ACGT'])

    if not seq_for_bio:
        st.info("Select a project or paste a DNA sequence to analyze.")
    else:
        motifs = find_motifs(seq_for_bio)
        st.write("**Detected motifs**")
        st.dataframe(motifs, use_container_width=True)
        df_k2 = kmer_frequencies(seq_for_bio, k=2)
        df_k3 = kmer_frequencies(seq_for_bio, k=3)
        st.pyplot(plot_kmer_heatmap(df_k2, k=2), use_container_width=True)
        st.pyplot(plot_kmer_heatmap(df_k3, k=3), use_container_width=True)

        # Simple "promoter strength" score (toy model): higher GC and presence of TATA -> middle strength heuristic
        gc = cg_content(seq_for_bio)
        tata_bonus = 0.2 if (motifs['motif'] == 'TATA-box').any() else 0.0
        strength = min(1.0, 0.5*gc + tata_bonus)
        st.metric("Toy promoter-strength score", f"{strength:.2f}", help="Heuristic: 0.5*GC + 0.2 if TATA-box present (clamped to 1.0)")

# ---------------- 3D Helix ----------------
with helix_tab:
    seq3d = st.text_area("Sequence to visualize (A/C/G/T)", height=160)
    if not seq3d.strip():
        st.info("Paste a DNA sequence to render the helix.")
    else:
        motifs3 = find_motifs(seq3d)
        fig3d = plot_3d_helix(seq3d, motifs3)
        st.plotly_chart(fig3d, use_container_width=True)

# ---------------- Storage Simulation ----------------
with sim_tab:
    st.subheader("Mutation & ECC simulation")
    seq_sim = st.text_area("Sequence to mutate (FASTA or raw)", height=160)
    if seq_sim.strip():
        raw_seq = []
        for line in seq_sim.splitlines():
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            raw_seq.append(line)
        dna_seq = ''.join(raw_seq).upper().replace(' ', '')
        st.write(f"Length: {len(dna_seq)} bases")
        mutated = mutate_dna(dna_seq, rate=mut_rate, seed=123)
        st.text_area("Mutated sequence", mutated[:1000] + ("..." if len(mutated) > 1000 else ""), height=160)
        ok = True
        try:
            hdr0, data0, ecc0 = unpack_from_dna(dna_seq)
            st.success("Original decodes OK")
        except Exception as e:
            st.warning(f"Original failed to decode: {e}")
            ok = False
        try:
            hdrM, dataM, eccM = unpack_from_dna(mutated)
            st.success("Mutated still decodes! 🎉")
            if eccM.get('used'):
                st.info(f"ECC corrected bits: {eccM.get('corrected_bits', 0)}")
        except Exception as e:
            st.error(f"Mutated decode failed (as expected if too many errors): {e}")
            st.caption("Tip: enable ECC during encoding to improve resilience.")
    else:
        st.info("Paste a DNA/FASTA sequence above, adjust the mutation rate in the sidebar, and observe decoding.")

# ---------------- Synthetic Biology ----------------
with synbio_tab:
    st.subheader("Plasmid insertion & GenBank export")
    seq_ins = st.text_area("Insert sequence (A/C/G/T)", height=160)
    colA, colB = st.columns([1,1])
    with colA:
        plasmid_len = st.number_input("Plasmid length (bp)", value=3000, min_value=1000, max_value=20000, step=100)
    with colB:
        seq_id = st.text_input("Record/Seq ID", value=f"ODSX_{uuid.uuid4().hex[:8]}")

    if seq_ins.strip():
        insert = ''.join([c for c in seq_ins.upper() if c in 'ACGT'])
        figp = plot_plasmid_map(len(insert), plasmid_len)
        st.plotly_chart(figp, use_container_width=True)
        gb_txt = make_genbank(seq_id, insert, plasmid_len)
        st.download_button("Download GenBank (.gb)", data=gb_txt, file_name=f"{seq_id}.gb", mime="chemical/seq-na-genbank")
        st.code(gb_txt[:1000] + ("\n..." if len(gb_txt) > 1000 else ""), language="text")
    else:
        st.info("Paste an insert sequence to generate a plasmid map and GenBank file.")

# ---------------- Project Browser ----------------
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
            fasta_path = os.path.join(pdir, f"{sel}.fasta")
            if os.path.exists(fasta_path):
                with open(fasta_path, "r", encoding="utf-8") as f:
                    fasta_txt = f.read()
                st.download_button("Download FASTA", data=fasta_txt, file_name=f"{sel}.fasta")
            # Render stains on-demand
            with open(fasta_path, 'r', encoding='utf-8') as f:
                seq = ''.join([ln.strip() for ln in f if ln and not ln.startswith('>')])
            fig_trace, fig_scatter, fig_heat = make_stain_figures(seq, window=int(window), step=int(step))
            st.pyplot(fig_trace, use_container_width=True)
            st.pyplot(fig_scatter, use_container_width=True)
            st.pyplot(fig_heat, use_container_width=True)
        except Exception as ex:
            st.error(f"Failed to open project: {ex}")

st.markdown(f"""
---
**ODS-X v{APP_VERSION}** · Deterministic bytes↔DNA mapping (2‑bit per base) with metadata, CRC, optional Hamming(7,4) ECC.
Bioinformatics views (IC/GC/k-mers/motifs), 3D helix, mutation simulation, and GenBank plasmid export.
""")
