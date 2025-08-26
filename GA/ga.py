
import os
import zlib
import uuid
import json
import math
import time
import typing as T
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# App meta
# =============================
APP_VERSION = "2.0.0-bio"
STORE_DIR = "ods_store"
INDEX_PATH = os.path.join(STORE_DIR, "index.json")
FASTA_WRAP = 80

# =============================
# Encoding schemes
# =============================
# Default mapping
DEFAULT_ENCODING_SCHEME = {0b00: "A", 0b01: "C", 0b10: "G", 0b11: "T"}
DEFAULT_DECODING_SCHEME = {v: k for k, v in DEFAULT_ENCODING_SCHEME.items()}
MAGIC = b"ODS2\x00"

# =============================
# Utilities
# =============================
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

@dataclass
class ODSHeader:
    version: str
    project_id: str
    filename: str
    mimetype: str
    length: int
    compressed: bool
    crc32: int
    original_size: int
    compressed_size: int
    compression_ratio: float
    # NEW: GA / encoding metadata
    ga_used: bool
    mapping: T.Dict[str, str]     # maps "00","01","10","11" -> base
    interleave: int               # interleave block size in bytes
    seed: int                     # RNG seed used for scrambling

    def to_bytes(self) -> bytes:
        payload = json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")
        size = len(payload).to_bytes(4, "big")
        return MAGIC + size + payload

    @staticmethod
    def from_bytes(buf: bytes) -> T.Tuple["ODSHeader", int]:
        if not buf.startswith(MAGIC):
            raise ValueError("Not an ODS v2 payload (magic mismatch)")
        size = int.from_bytes(buf[5:9], "big")
        js = json.loads(buf[9:9+size].decode("utf-8"))
        hdr = ODSHeader(**js)
        return hdr, 9 + size

def detect_mime(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in {".txt", ".md", ".csv"}: return "text/plain"
    if ext in {".png"}: return "image/png"
    if ext in {".jpg", ".jpeg"}: return "image/jpeg"
    if ext in {".gif"}: return "image/gif"
    if ext in {".mp3"}: return "audio/mpeg"
    if ext in {".wav"}: return "audio/wav"
    if ext in {".mp4"}: return "video/mp4"
    return "application/octet-stream"

# =============================
# Core conversion
# =============================
def bytes_to_dna(data: bytes, mapping: T.Dict[int, str]) -> str:
    # mapping is like {0:'A',1:'C',2:'G',3:'T'}
    dna_chars = []
    for b in data:
        dna_chars.append(mapping[(b >> 6) & 0b11])
        dna_chars.append(mapping[(b >> 4) & 0b11])
        dna_chars.append(mapping[(b >> 2) & 0b11])
        dna_chars.append(mapping[b & 0b11])
    return "".join(dna_chars)

def dna_to_bytes(seq: str, revmap: T.Dict[str, int]) -> bytes:
    if len(seq) % 4 != 0:
        raise ValueError("DNA length must be a multiple of 4 for byte-aligned decoding")
    allowed = set(revmap.keys())
    if any(ch not in allowed for ch in seq):
        raise ValueError("DNA contains invalid symbols")
    out = bytearray()
    for i in range(0, len(seq), 4):
        quad = seq[i:i+4]
        b = (revmap[quad[0]] << 6) | (revmap[quad[1]] << 4) | (revmap[quad[2]] << 2) | revmap[quad[3]]
        out.append(b)
    return bytes(out)

def to_fasta(seq_id: str, dna: str, wrap: int = FASTA_WRAP) -> str:
    lines = [f">{seq_id}"]
    for i in range(0, len(dna), wrap):
        lines.append(dna[i:i+wrap])
    return "\n".join(lines) + "\n"

# =============================
# Bio: analytics on DNA-like sequence
# =============================
NUCS = np.array(list("ACGT"))

def sliding_windows(seq: str, window: int, step: int) -> T.Iterable[str]:
    if window > len(seq) and len(seq) > 0:
        yield seq
        return
    for i in range(0, max(0, len(seq) - window + 1), step):
        yield seq[i:i+window]

def index_of_coincidence(window_seq: str) -> float:
    N = len(window_seq)
    if N <= 1:
        return 0.0
    counts = np.fromiter((window_seq.count(b) for b in "ACGT"), dtype=np.int64, count=4)
    num = np.sum(counts * (counts - 1))
    den = N * (N - 1)
    return float(num) / float(den) if den else 0.0

def cg_content(window_seq: str) -> float:
    if not window_seq:
        return 0.0
    c = window_seq.count("C")
    g = window_seq.count("G")
    return (c + g) / len(window_seq)

def dinuc_matrix(seq: str) -> np.ndarray:
    idx = {c:i for i,c in enumerate("ACGT")}
    counts = np.zeros((4,4), dtype=np.float64)
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    s = counts.sum()
    return counts / s if s > 0 else counts

def cpg_islands(seq: str, window: int = 200, ooe_thresh: float = 0.6) -> T.List[T.Tuple[int,int,float]]:
    """Return intervals (start,end,oe) where CpG obs/exp exceeds threshold.
       oe = (#CpG * len) / (#C * #G)
    """
    out = []
    n = len(seq)
    for i in range(0, n - window + 1, max(50, window//4)):
        s = seq[i:i+window]
        c = s.count("C"); g = s.count("G"); cpg = sum(1 for j in range(len(s)-1) if s[j:j+2] == "CG")
        if c and g:
            oe = (cpg * len(s)) / (c * g)
            if oe >= ooe_thresh and (c+g)/len(s) >= 0.5:
                out.append((i, i+window, round(oe, 3)))
    return out

CODON_TABLE = {
    # Simplified standard table (start/stop not differentiated for speed here)
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}

def find_orfs(seq: str, min_len: int = 90) -> T.List[dict]:
    stops = {"TAA","TAG","TGA"}
    res = []
    def scan(s: str, strand: int):
        n = len(s)
        for frame in range(3):
            i = frame
            while i+3 <= n:
                cod = s[i:i+3]
                if cod == "ATG":
                    j = i + 3
                    while j+3 <= n and s[j:j+3] not in stops:
                        j += 3
                    length = j - i
                    if length >= min_len:
                        res.append({"start": i if strand==+1 else n-(i+length),
                                    "end": j if strand==+1 else n-(i),
                                    "length": length,
                                    "frame": frame if strand==+1 else -(frame+1),
                                    "strand": strand})
                    i = j + 3
                else:
                    i += 3
    scan(seq, +1)
    # reverse complement
    rc_map = str.maketrans("ACGT", "TGCA")
    rc = seq.translate(rc_map)[::-1]
    scan(rc, -1)
    return sorted(res, key=lambda d: -d["length"])[:200]

def codon_usage(seq: str) -> pd.DataFrame:
    counts = {c:0 for c in CODON_TABLE.keys()}
    n = len(seq)
    for i in range(0, n - 2, 3):
        cod = seq[i:i+3]
        if cod in counts:
            counts[cod] += 1
    df = pd.DataFrame([{"codon":k, "aa":CODON_TABLE[k], "count":v} for k,v in counts.items()])
    df["freq"] = df["count"] / max(1, df["count"].sum())
    return df.sort_values("codon")

def make_stain_plots(seq: str, window: int = 300, step: int = 30, project_dir: str = ".") -> dict:
    os.makedirs(project_dir, exist_ok=True)
    ic_values = []
    cg_values = []
    for w in sliding_windows(seq, window, step):
        ic_values.append(index_of_coincidence(w))
        cg_values.append(cg_content(w))
    if len(ic_values) == 0:
        ic_values = [0.0]; cg_values = [0.0]
    x = np.arange(len(ic_values))
    fig1 = plt.figure(figsize=(9, 3))
    plt.plot(x, ic_values, label="IC")
    plt.plot(x, cg_values, label="CG fraction")
    plt.title("ODS – IC & CG% over sliding windows")
    plt.xlabel(f"Window index (window={window}, step={step})")
    plt.ylabel("Value")
    plt.legend()
    path1 = os.path.join(project_dir, "trace_ic_cg.png")
    fig1.tight_layout(); fig1.savefig(path1, dpi=150); plt.close(fig1)

    fig2 = plt.figure(figsize=(4, 4))
    plt.scatter(cg_values, ic_values, s=6)
    plt.title("ODS – IC vs CG")
    plt.xlabel("CG fraction")
    plt.ylabel("Index of Coincidence")
    path2 = os.path.join(project_dir, "scatter_ic_vs_cg.png")
    fig2.tight_layout(); fig2.savefig(path2, dpi=150); plt.close(fig2)

    counts = dinuc_matrix(seq)
    fig3 = plt.figure(figsize=(4.5, 4))
    plt.imshow(counts, aspect='equal')
    plt.xticks(range(4), list("ACGT"));
    plt.yticks(range(4), list("ACGT"))
    plt.title("ODS – Dinucleotide heatmap")
    plt.colorbar(label="Probability")
    path3 = os.path.join(project_dir, "heatmap_dinuc.png")
    fig3.tight_layout(); fig3.savefig(path3, dpi=150); plt.close(fig3)

    # CpG islands quick plot
    islands = cpg_islands(seq)
    fig4 = plt.figure(figsize=(9,2.2))
    xs = [ (a+b)/2 for a,b,_ in islands ]
    ys = [ oe for _,_,oe in islands ]
    plt.scatter(xs, ys, s=10)
    plt.title("CpG islands (center vs O/E)")
    plt.xlabel("Position"); plt.ylabel("CpG O/E")
    path4 = os.path.join(project_dir, "cpg_islands.png")
    fig4.tight_layout(); fig4.savefig(path4, dpi=150); plt.close(fig4)

    return {"trace": path1, "scatter": path2, "heatmap": path3, "cpg": path4}

# =============================
# Genetic Algorithm optimizer
# =============================
def score_dna(seq: str, target_gc: float = 0.5, homopoly_penalty: float = 3.0, window: int = 200):
    """Lower is better."""
    if not seq:
        return 1e9
    # GC deviation (windowed)
    dev = 0.0
    step = max(50, window//4)
    for w in sliding_windows(seq, window, step):
        gc = cg_content(w)
        dev += (gc - target_gc)**2
    dev /= max(1, len(seq)//step)

    # homopolymer penalty
    max_run = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1; max_run = max(max_run, cur)
        else:
            cur = 1
    homopen = homopoly_penalty * max(0, max_run - 4)  # allow up to 4 safely

    # dinucleotide evenness (prefer flatter distribution)
    mat = dinuc_matrix(seq)
    flat = 0.25 * 0.25
    even = np.mean((mat - flat)**2)

    return dev + homopen + even

def scramble_bytes(data: bytes, seed: int, interleave: int) -> bytes:
    if interleave <= 1:
        return data
    rng = np.random.default_rng(seed)
    n = len(data)
    out = bytearray(n)
    for ofs in range(0, n, interleave):
        chunk = bytearray(data[ofs:ofs+interleave])
        idx = np.arange(len(chunk))
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            out[ofs+i] = chunk[j]
    return bytes(out)

def ga_optimize_mapping(data: bytes,
                        pop_size: int = 12,
                        iters: int = 12,
                        interleave_options=(1, 64, 128),
                        seed: int = None):
    """Evolve 2-bit->base mapping and interleave to minimize score_dna."""
    seed = int(time.time()*1000) % (2**32-1) if seed is None else seed
    rng = np.random.default_rng(seed)

    # All 24 permutations of "ACGT"
    from itertools import permutations
    perms = list(permutations("ACGT"))
    pop_idx = rng.choice(len(perms), size=pop_size, replace=False)
    pop_maps = [ {i:b for i,b in enumerate(perms[j])} for j in pop_idx ]
    pop_inter = list(rng.choice(interleave_options, size=pop_size))

    def fitness(mapping, inter):
        scrambled = scramble_bytes(data, seed, inter)
        dna = bytes_to_dna(scrambled, mapping)
        return score_dna(dna)

    scores = np.array([fitness(m, it) for m,it in zip(pop_maps, pop_inter)])
    for _ in range(iters):
        # select top half
        idx = np.argsort(scores)[:max(2, pop_size//2)]
        elites = [pop_maps[i] for i in idx]
        elites_inter = [pop_inter[i] for i in idx]

        # mutate to refill
        new_maps = elites.copy()
        new_inter = elites_inter.copy()
        while len(new_maps) < pop_size:
            base = elites[rng.integers(len(elites))].copy()
            # swap two outputs
            a,b = rng.choice([0,1,2,3], size=2, replace=False)
            base[a], base[b] = base[b], base[a]
            new_maps.append(base)
            new_inter.append(int(rng.choice(interleave_options)))
        pop_maps, pop_inter = new_maps, new_inter
        scores = np.array([fitness(m, it) for m,it in zip(pop_maps, pop_inter)])

    best_i = int(np.argmin(scores))
    return pop_maps[best_i], int(pop_inter[best_i]), seed, float(scores[best_i])

# =============================
# Pack/Unpack with GA
# =============================
def pack_to_dna(name: str, data: bytes, allow_compression: bool, use_ga: bool) -> T.Tuple[ODSHeader, str]:
    mimetype = detect_mime(name)
    original_size = len(data)

    # Try compression first for better channel utilization
    compressed_data = zlib.compress(data) if allow_compression else data
    compressed_size = len(compressed_data)
    use_compressed = allow_compression and compressed_size < original_size
    payload = compressed_data if use_compressed else data

    # GA optimize mapping & interleave
    if use_ga and len(payload) > 0:
        mapping, interleave, seed, _ = ga_optimize_mapping(payload)
    else:
        mapping = {i:b for i,b in DEFAULT_ENCODING_SCHEME.items()}
        interleave, seed = 1, 0

    scrambled = scramble_bytes(payload, seed, interleave)
    dna = bytes_to_dna(scrambled, mapping)

    crc = zlib.crc32(payload) & 0xFFFFFFFF  # checksum on the true payload
    ratio = round(compressed_size / original_size, 3) if original_size > 0 else 1.0

    hdr = ODSHeader(
        version=APP_VERSION,
        project_id=str(uuid.uuid4()),
        filename=name,
        mimetype=mimetype,
        length=len(payload),
        compressed=use_compressed,
        crc32=crc,
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=ratio,
        ga_used=use_ga,
        mapping={format(k, "02b"): v for k, v in mapping.items()},
        interleave=int(interleave),
        seed=int(seed)
    )
    packet = hdr.to_bytes() + payload  # store *payload* after header (not scrambled), so decoding is robust
    # For storage/transport we still output optimized DNA for "digital staining"
    return hdr, dna

def unpack_from_dna(dna: str, hdr_only: bool = False) -> T.Tuple[ODSHeader, bytes]:
    # The DNA users paste may be optimized form; header+payload are inside the DNA bytes
    packet = dna_to_bytes(dna, {v:k for k,v in DEFAULT_ENCODING_SCHEME.items()})
    hdr, ofs = ODSHeader.from_bytes(packet)
    if hdr_only:
        return hdr, b""
    payload = packet[ofs:ofs + hdr.length]
    if len(payload) != hdr.length:
        raise ValueError("Payload truncated")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != hdr.crc32:
        raise ValueError("CRC mismatch – data corrupted or wrong sequence")
    data = zlib.decompress(payload) if hdr.compressed else payload
    return hdr, data

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="ODS Bio+GA", layout="wide")
st.title("🧬 ODS Bio + GA")
st.caption("Encode files to DNA-like sequences with genetic algorithm optimization, and analyze sequences with bioinformatics tools.")

with st.sidebar:
    st.header("1) Encode")
    uploaded = st.file_uploader("Upload any file", type=None)
    allow_compression = st.checkbox("Try to compress payload", value=True)
    use_ga = st.checkbox("Genetic Algorithm optimize encoding", value=True)
    window = st.number_input("Sliding window length", value=300, min_value=50, step=10)
    step = st.number_input("Window step", value=30, min_value=1, step=1)
    do_encode = st.button("Encode + Analyze")

    st.header("2) Decode")
    seq_input = st.text_area("Paste DNA sequence (FASTA or raw)", height=140)
    do_decode = st.button("Decode DNA back to file")

ensure_store()
col1, col2 = st.columns(2)

# ENCODE
if do_encode and uploaded is not None:
    name = uploaded.name
    raw = uploaded.read()
    if raw is None:
        st.error("Empty file")
    else:
        hdr, dna = pack_to_dna(name, raw, allow_compression, use_ga)
        project_dir = os.path.join(STORE_DIR, hdr.project_id)
        os.makedirs(project_dir, exist_ok=True)

        # Save FASTA (optimized DNA)
        fasta = to_fasta(hdr.project_id, dna)
        fasta_path = os.path.join(project_dir, f"{hdr.project_id}.fasta")
        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(fasta)

        # Save meta
        meta = asdict(hdr)
        meta.update({"dna_length": len(dna)})
        with open(os.path.join(project_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Plots & analytics
        paths = make_stain_plots(dna, window=int(window), step=int(step), project_dir=project_dir)

        # CpG islands & ORFs & codon usage
        islands = cpg_islands(dna)
        orfs = find_orfs(dna, min_len=90)
        cu_df = codon_usage(dna)

        # Index
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
            st.image(paths["cpg"], caption="CpG islands", use_column_width=True)

        st.markdown("### 🧪 Bioinformatics Tables")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**CpG islands (start, end, O/E)**")
            if islands:
                df_islands = pd.DataFrame(islands, columns=["start","end","CpG_OE"])
                st.dataframe(df_islands, use_container_width=True)
            else:
                st.info("No CpG-like islands detected at current thresholds.")
        with c2:
            st.write("**Top ORFs (by length)**")
            if orfs:
                st.dataframe(pd.DataFrame(orfs), use_container_width=True)
            else:
                st.info("No long ORFs found.")

        st.markdown("### 🧬 Codon usage")
        st.dataframe(cu_df, use_container_width=True)

# DECODE
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

# Stored projects
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
            for img in ["trace_ic_cg.png", "scatter_ic_vs_cg.png", "heatmap_dinuc.png", "cpg_islands.png"]:
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
**ODS Bio v{APP_VERSION}** · 2-bit DNA mapping optimized with a Genetic Algorithm (homopolymer/GC/dinucleotide score).
Bio analytics: CpG islands, ORFs (6-frame), codon usage, IC/CG traces, dinucleotide heatmap.
""")
