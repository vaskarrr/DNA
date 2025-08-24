import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter

"""
Closed-loop JPEG-DNA prototype with robustness features:
✔ Restart markers
✔ Fixed-length length fields
✔ CRC-16 per block
✔ Optional repetition ECC
✔ Soft-decode CRC-failed blocks (optional)
✔ Corruption visualization
✔ Restart interval resets DC predictor
"""

#############################
# Math helpers: 2D DCT / IDCT
#############################

def dct_1d(v):
    N = v.shape[0]
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    alpha = np.ones((N, 1))
    alpha[0] = 1 / np.sqrt(2)
    C = np.sqrt(2 / N) * alpha * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return C @ v


def idct_1d(V):
    N = V.shape[0]
    k = np.arange(N)[None, :]
    n = np.arange(N)[:, None]
    alpha = np.ones((N, 1))
    alpha[0] = 1 / np.sqrt(2)
    C = np.sqrt(2 / N) * alpha.T * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return C @ V


def dct2(block):
    return dct_1d(dct_1d(block.T).T)


def idct2(block):
    return idct_1d(idct_1d(block.T).T)

#############################
# JPEG helpers
#############################
ZIGZAG_IDX = np.array([
    [0, 1, 5, 6, 14, 15, 27, 28],
    [2, 4, 7, 13, 16, 26, 29, 42],
    [3, 8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63],
]).flatten()

STD_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.int32)


def quality_scale(Q):
    Q = max(1, min(100, int(Q)))
    scale = 5000 / Q if Q < 50 else 200 - 2 * Q
    table = np.floor((STD_LUMA_Q * scale + 50) / 100).astype(np.int32)
    table[table == 0] = 1
    return table


def blockify(img, block=8):
    h, w = img.shape
    H = (h + block - 1) // block * block
    W = (w + block - 1) // block * block
    padded = np.zeros((H, W), dtype=np.float32)
    padded[:h, :w] = img
    blocks = [padded[y:y + block, x:x + block] for y in range(0, H, block) for x in range(0, W, block)]
    return np.array(blocks), H, W


def deblockify(blocks, H, W, block=8):
    out = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y in range(0, H, block):
        for x in range(0, W, block):
            out[y:y + block, x:x + block] = blocks[idx]
            idx += 1
    return out


def zigzag(block):
    return block.flatten()[ZIGZAG_IDX]


def izigzag(vec):
    block = np.zeros(64, dtype=np.float32)
    block[ZIGZAG_IDX] = vec
    return block.reshape(8, 8)

#############################
# Categories & Huffman
#############################
DNA_CATS = [
    (0, 0, 0, 0),
    (2, 2, 1, 5),
    (3, 3, 6, 25),
    (4, 4, 26, 75),
    (5, 5, 76, 275),
    (6, 6, 276, 775),
    (7, 7, 776, 2775),
    (8, 8, 2776, 7775),
]
CAT_BY_ABS = [(lo, hi, cid, nts) for (cid, nts, lo, hi) in DNA_CATS if cid != 0]
EOB = (0, 0)
ZRL = (15, 0)


class TernaryNode:
    def __init__(self, sym=None, freq=0, children=None):
        self.sym = sym
        self.freq = freq
        self.children = children or []

    def is_leaf(self):
        return self.sym is not None


def build_ternary_huffman(freqs):
    nodes = [TernaryNode(sym=s, freq=f) for s, f in freqs.items() if f > 0]
    if not nodes:
        return {}, TernaryNode()
    while len(nodes) % 2 == 0:
        nodes.append(TernaryNode(sym=None, freq=0))
    while len(nodes) > 1:
        nodes.sort(key=lambda n: n.freq)
        a, b, c = nodes[:3]
        parent = TernaryNode(children=[a, b, c], freq=a.freq + b.freq + c.freq)
        nodes = nodes[3:] + [parent]
    root = nodes[0]
    codes = {}

    def walk(node, prefix):
        if node.is_leaf() and node.sym is not None:
            codes[node.sym] = prefix or [0]
            return
        for i, ch in enumerate(node.children):
            walk(ch, prefix + [i])

    walk(root, [])
    return codes, root


NEXT_MAP = {
    None: ['A', 'C', 'G'],
    'A': ['C', 'G', 'T'],
    'C': ['A', 'G', 'T'],
    'G': ['A', 'C', 'T'],
    'T': ['A', 'C', 'G'],
}


def trits_to_dna(trits, start_base=None):
    seq = []
    prev = start_base
    for t in trits:
        choices = NEXT_MAP[prev]
        base = choices[t % 3]
        if len(seq) >= 3 and seq[-1] == seq[-2] == seq[-3] == base:
            base = choices[(t + 1) % 3]
        seq.append(base)
        prev = base
    return ''.join(seq)


def dna_to_trits(seq, start_base=None):
    trits = []
    prev = start_base
    for ch in seq:
        choices = NEXT_MAP[prev]
        t = choices.index(ch) if ch in choices else 0
        trits.append(t)
        prev = ch
    return trits

#############################
# Value coding & helpers
#############################

def value_to_category(v):
    if v == 0:
        return 0, 0, 0, 0
    a = abs(v)
    for lo, hi, cid, nts in CAT_BY_ABS:
        if lo <= a <= hi:
            return cid, nts, lo, hi
    lo, hi, cid, nts = CAT_BY_ABS[-1]
    return cid, nts, lo, hi


def int_to_trits(n, length):
    return [(n // 3 ** i) % 3 for i in reversed(range(length))]


def trits_to_int(trits):
    return sum(t * 3 ** i for i, t in enumerate(trits[::-1]))


def encode_value_paircode(v, nts_len, lo, hi, start_base):
    idx = (v - lo) if v > 0 else (hi - lo + 1) + (abs(v) - lo)
    idx = min(idx, 3 ** nts_len - 1)
    trits = int_to_trits(idx, nts_len)
    return trits_to_dna(trits, start_base)


def decode_value_paircode(dna, nts_len, lo, hi, start_base):
    trits = dna_to_trits(dna[:nts_len], start_base)
    idx = trits_to_int(trits)
    mid = (hi - lo + 1)
    return lo + idx if idx < mid else -(lo + (idx - mid))

#############################
# JPEG-like encode
#############################

def jpeg_like_encode(img_gray, quality=60):
    img = img_gray.astype(np.float32) - 128.0
    blocks, H, W = blockify(img, 8)
    q = quality_scale(quality).astype(np.float32)
    coeffs = [np.round(dct2(b) / q) for b in blocks]
    dc_vals = []
    ac_tuples = []
    prev_dc = 0
    for c in coeffs:
        zz = zigzag(c)
        dc = int(zz[0])
        dc_diff = dc - prev_dc
        prev_dc = dc
        dc_vals.append(dc_diff)
        ac = zz[1:]
        run = 0
        block_ac = []
        for v in ac:
            v = int(v)
            if v == 0:
                run += 1
                if run == 16:
                    block_ac.append((15, 0))
                    run = 0
            else:
                cid, nts_len, lo, hi = value_to_category(v)
                block_ac.append((run, cid))
                block_ac.append(('VAL', v))
                run = 0
        block_ac.append((0, 0))
        ac_tuples.append(block_ac)
    return dc_vals, ac_tuples, H, W, q


def collect_symbols(dc_vals, ac_tuples):
    dc_syms = [('DC', value_to_category(v)[0]) for v in dc_vals]
    ac_syms = []
    for L in ac_tuples:
        for item in L:
            if item == (0, 0):
                ac_syms.append(('EOB',))
            elif item == (15, 0):
                ac_syms.append(('ZRL',))
            elif isinstance(item, tuple) and item[0] != 'VAL':
                ac_syms.append(('AC', item[0], item[1]))
    return dc_syms, ac_syms

#############################
# ECC, framing & CRC
#############################
RST_MARKER = 'ACGTACGTACGTACGTACGTACGT'
LEN_TRITS = 20
CRC_TRITS = 12


def crc16(data, poly=0x1021, init=0xFFFF):
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def header_pack_length(length_nt):
    return trits_to_dna(int_to_trits(length_nt, LEN_TRITS))


def header_unpack_length(dna):
    return trits_to_int(dna_to_trits(dna[:LEN_TRITS]))


def crc_pack_trits(crc_val):
    return trits_to_dna(int_to_trits(crc_val, CRC_TRITS))


def crc_unpack_trits(dna):
    return trits_to_int(dna_to_trits(dna[:CRC_TRITS]))


def replicate_ecc(seq, rep=1):
    return ''.join(seq for _ in range(rep)) if rep > 1 else seq


def majority_vote(strings):
    if not strings:
        return ''
    L = min(len(s) for s in strings)
    strings = [s[:L] for s in strings]
    out = []
    for i in range(L):
        col = [s[i] for s in strings]
        out.append(Counter(col).most_common(1)[0][0])
    return ''.join(out)


def encode_block_payload(bi, dc_val, ac_list, dc_codes, ac_codes):
    prev_base = None
    payload = ''
    cid, nts_len, lo, hi = value_to_category(dc_val)
    dna_cat = trits_to_dna(dc_codes[('DC', cid)], prev_base)
    payload += dna_cat
    prev_base = dna_cat[-1] if dna_cat else None
    if cid != 0:
        dna_val = encode_value_paircode(dc_val, nts_len, lo, hi, prev_base)
        payload += dna_val
        prev_base = dna_val[-1] if dna_val else prev_base
    for item in ac_list:
        if item == (0, 0):
            tr = ac_codes[('EOB',)]
        elif item == (15, 0):
            tr = ac_codes[('ZRL',)]
        elif isinstance(item, tuple) and item[0] != 'VAL':
            tr = ac_codes[('AC', item[0], item[1])]
        else:
            tr = None
        if tr is not None:
            dna = trits_to_dna(tr, prev_base)
            payload += dna
            prev_base = dna[-1] if dna else prev_base
        if isinstance(item, tuple) and item[0] == 'VAL':
            v = item[1]
            cid2, nts2, lo2, hi2 = value_to_category(v)
            if cid2 != 0:
                dna_val = encode_value_paircode(v, nts2, lo2, hi2, prev_base)
                payload += dna_val
                prev_base = dna_val[-1]
    return payload


def encode_stream(dc_vals, ac_lists, dc_codes, ac_codes, restart_interval=8, ecc_rep=1):
    frames = []
    for i in range(len(dc_vals)):
        payload = encode_block_payload(i, dc_vals[i], ac_lists[i], dc_codes, ac_codes)
        crc = crc16(payload.encode('ascii'))
        header_len = header_pack_length(len(payload) * ecc_rep)
        crc_dna = crc_pack_trits(crc)
        payload_ecc = replicate_ecc(payload, ecc_rep)
        frame = RST_MARKER + header_len + payload_ecc + crc_dna
        frames.append(frame)
    return ''.join(frames)


def decode_stream(dna_stream, blocks_count, dc_codes, ac_codes, restart_interval, ecc_rep, soft_decode):
    def rebuild_tree(codes):
        root = TernaryNode()
        for sym, trits in codes.items():
            node = root
            for t in trits:
                while len(node.children) <= t:
                    node.children.append(TernaryNode())
                node = node.children[t]
            node.sym = sym
        return root

    dc_root = rebuild_tree(dc_codes)
    ac_root = rebuild_tree(ac_codes)

    def read_sym(dna, pos, root, start_base):
        node = root
        prev = start_base
        while True:
            if pos >= len(dna):
                return None, pos, prev
            ch = dna[pos]
            choices = NEXT_MAP[prev]
            t = choices.index(ch) if ch in choices else 0
            if t >= len(node.children):
                node.children += [TernaryNode()] * (t - len(node.children) + 1)
            node = node.children[t]
            prev = ch
            pos += 1
            if node.sym is not None:
                return node.sym, pos, prev

    frames = []
    i = 0
    while True:
        idx = dna_stream.find(RST_MARKER, i)
        if idx == -1:
            break
        start = idx + len(RST_MARKER)
        if start + LEN_TRITS > len(dna_stream):
            break
        length_nt = header_unpack_length(dna_stream[start:start + LEN_TRITS])
        pay_start = start + LEN_TRITS
        pay_end = pay_start + length_nt
        crc_start = pay_end
        crc_end = crc_start + CRC_TRITS
        if crc_end > len(dna_stream):
            break
        payload_ecc = dna_stream[pay_start:pay_end]
        crc_val = crc_unpack_trits(dna_stream[crc_start:crc_end])
        if ecc_rep > 1:
            chunk_len = length_nt // ecc_rep
            reps = [payload_ecc[j * chunk_len:(j + 1) * chunk_len] for j in range(ecc_rep)]
            payload = majority_vote(reps)
        else:
            payload = payload_ecc
        crc_ok = (crc16(payload.encode('ascii')) == crc_val)
        if crc_ok or soft_decode:
            frames.append((crc_ok, payload))
        else:
            frames.append((False, ''))
        i = crc_end
        if len(frames) >= blocks_count:
            break

    dc_vals = []
    ac_lists = []
    corruption_mask = np.ones(blocks_count, dtype=bool)
    for bi in range(blocks_count):
        ok, payload = frames[bi] if bi < len(frames) else (False, '')
        corruption_mask[bi] = not ok
        prev_base = None
        pos = 0
        sym, pos, prev_base = read_sym(payload, pos, dc_root, prev_base)
        if sym is None or sym[0] != 'DC':
            dc_vals.append(0)
            ac_lists.append([(0, 0)])
            continue
        cid = sym[1]
        if cid == 0:
            dc_v = 0
        else:
            nts_len = cid
            lo, hi = next(((lo, hi) for (lo, hi, c, _) in CAT_BY_ABS if c == cid), (1, 1))
            dna_val = payload[pos:pos + nts_len]
            dc_v = decode_value_paircode(dna_val, nts_len, lo, hi, prev_base) if len(dna_val) >= nts_len else 0
            pos += nts_len
            prev_base = dna_val[-1] if dna_val else prev_base
        dc_vals.append(dc_v)
        block_ac = []
        guard = 0
        while pos < len(payload) and guard < 10000:
            guard += 1
            sym, pos, prev_base = read_sym(payload, pos, ac_root, prev_base)
            if sym is None:
                break
            if sym == ('EOB',):
                block_ac.append((0, 0))
                break
            elif sym == ('ZRL',):
                block_ac.append((15, 0))
            elif sym[0] == 'AC':
                run, rcid = sym[1], sym[2]
                block_ac.append((run, rcid))
                if rcid != 0:
                    nts_len = rcid
                    lo, hi = next(((lo, hi) for (lo, hi, c, _) in CAT_BY_ABS if c == rcid), (1, 1))
                    dna_val = payload[pos:pos + nts_len]
                    val = decode_value_paircode(dna_val, nts_len, lo, hi, prev_base) if len(dna_val) >= nts_len else 0
                    pos += nts_len
                    prev_base = dna_val[-1] if dna_val else prev_base
                    block_ac.append(('VAL', val))
        if not block_ac or block_ac[-1] != (0, 0):
            block_ac.append((0, 0))
        ac_lists.append(block_ac)
    return dc_vals, ac_lists, corruption_mask

#############################
# Rebuild image from DC/AC
#############################

def rebuild_blocks(dc_vals, ac_lists, H, W, q, restart_interval):
    blocks_n = len(dc_vals)
    blocks = []
    prev_dc = 0
    for i in range(blocks_n):
        if restart_interval > 0 and i % restart_interval == 0:
            prev_dc = 0
        dc = prev_dc + dc_vals[i]
        prev_dc = dc
        coef = np.zeros(64, dtype=np.float32)
        coef[0] = dc
        ac = ac_lists[i]
        k = 1
        j = 0
        while j < len(ac):
            item = ac[j]
            if item == (0, 0):
                break
            if item == (15, 0):
                k += 16
                j += 1
                continue
            if isinstance(item, tuple) and item[0] != 'VAL':
                run, cid = item
                k += run
                j += 1
                val = 0
                if j < len(ac) and isinstance(ac[j], tuple) and ac[j][0] == 'VAL':
                    val = ac[j][1]
                if k < 64:
                    coef[k] = val
                k += 1
            j += 1
        block = izigzag(coef)
        blocks.append(block)
    blocks = np.array(blocks)
    out_blocks = []
    qf = q.astype(np.float32)
    for b in blocks:
        deq = b * qf
        img_b = idct2(deq)
        out_blocks.append(img_b)
    out = deblockify(np.array(out_blocks), H, W, 8)
    out = np.clip(out + 128.0, 0, 255).astype(np.uint8)
    return out

#############################
# DNA utils
#############################

def gc_content(seq):
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in 'GC')
    return 100.0 * gc / len(seq)


def max_homopolymer(seq):
    if not seq:
        return 0
    m = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            m = max(m, cur)
        else:
            cur = 1
    return m


def split_into_oligos(seq, oligo_len=220, tag_prefix='OLG'):
    oligos = []
    i = 0
    idx = 0
    while i < len(seq):
        chunk = seq[i:i + oligo_len]
        header = f"{tag_prefix}_{idx:06d}"
        oligos.append((header, chunk))
        i += oligo_len
        idx += 1
    return oligos

#############################
# Streamlit UI
#############################
st.set_page_config(page_title="JPEG-DNA (Robust)", layout="wide")
st.title("🧬 JPEG-DNA (Closed-loop, Robust Prototype)")
st.caption("Restart markers • Framed blocks • Length fields • CRC-16 • Optional repetition ECC • Ternary Huffman + PAIRCODE-like values")

with st.sidebar:
    st.header("Parameters")
    quality = st.slider("JPEG Quality", 10, 95, 60, step=1)
    oligo_len = st.slider("Oligo length (nt)", 80, 300, 220, step=10)
    restart_interval = st.slider("Restart interval (blocks)", 1, 64, 8)
    ecc_rep = st.selectbox("Repetition ECC", [1, 3], index=0, help="3× repetition with majority vote per nt")
    soft_decode = st.checkbox("Soft-decode CRC-failed blocks", value=True)

uploaded = st.file_uploader("Upload image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns([1, 1])

if uploaded:
    img = Image.open(uploaded).convert('L')
    img_np = np.array(img)
    col1.image(img, caption=f"Original ({img_np.shape[1]}×{img_np.shape[0]})", use_column_width=True)

    # Encode to DC/AC
    dc_vals, ac_lists, H, W, q = jpeg_like_encode(img_np, quality)

    # Build ternary codes
    dc_syms, ac_syms = collect_symbols(dc_vals, ac_lists)
    dc_codes, ac_codes = build_ternary_huffman(Counter(dc_syms))[0], None
    # build_ternary_huffman returns (codes, root); reuse helper directly for both
    dc_codes, _ = build_ternary_huffman(Counter(dc_syms))
    ac_codes, _ = build_ternary_huffman(Counter(ac_syms))

    # Encode framed stream
    dna_stream = encode_stream(dc_vals, ac_lists, dc_codes, ac_codes, restart_interval=restart_interval, ecc_rep=ecc_rep)

    # Split to oligos & stats
    oligos = split_into_oligos(dna_stream, oligo_len=oligo_len)
    total_nt = len(dna_stream)
    gc = gc_content(dna_stream)
    hp = max_homopolymer(dna_stream)

    with col2:
        st.subheader("DNA Stream Stats")
        st.metric("Total nucleotides", f"{total_nt:,}")
        st.metric("GC content (%)", f"{gc:.2f}")
        st.metric("Max homopolymer length", f"{hp}")
        st.write(f"Oligos: {len(oligos)} (≈{oligo_len} nt each)")
        st.code(dna_stream[:300] + ("..." if len(dna_stream) > 300 else ""), language="text")

    st.divider()
    st.subheader("Decoding (framed, CRC-checked)")

    try:
        blocks_count = len(dc_vals)
        d_dc, d_ac, corr_mask = decode_stream(''.join(seq for _, seq in oligos), blocks_count, dc_codes, ac_codes,
                                              restart_interval=restart_interval, ecc_rep=ecc_rep, soft_decode=soft_decode)
        recon = rebuild_blocks(d_dc, d_ac, H, W, q, restart_interval)
        recon = recon[:img_np.shape[0], :img_np.shape[1]]

        # Overlay corrupted blocks
        overlay = recon.copy()
        block = 8
        bw = W // block
        for idx, corrupted in enumerate(corr_mask):
            if corrupted:
                y = (idx // bw) * block
                x = (idx % bw) * block
                overlay[y:y + block, x:x + block] = 128

        col1.image(overlay, caption="Decoded image (gray blocks = corrupted)", use_column_width=True)
        mse = float(np.mean((recon.astype(np.float32) - img_np.astype(np.float32)) ** 2))
        psnr = 10 * np.log10(255 * 255 / (mse + 1e-9))
        col1.write(f"PSNR vs original: **{psnr:.2f} dB** (compression only)")
    except Exception as e:
        st.warning(f"Decoding failed: {e}")

    st.subheader("Downloads")
    fasta = []
    for name, seq in oligos:
        fasta.append(f">{name}\n{seq}")
    fasta_txt = "\n".join(fasta)
    st.download_button("Download oligos (FASTA)", data=fasta_txt, file_name="oligos.fasta", mime="text/plain")
    st.download_button("Download raw DNA stream (TXT)", data=dna_stream, file_name="dna_stream.txt", mime="text/plain")

    st.subheader("Metadata")
    st.json({
        'blocks': len(dc_vals),
        'quality': quality,
        'restart_interval': restart_interval,
        'ecc_rep': ecc_rep,
        'soft_decode': soft_decode,
        'dc_code_symbols': len(dc_codes),
        'ac_code_symbols': len(ac_codes),
        'gc_percent': gc,
        'max_homopolymer': hp,
    })
else:
    st.info("Upload a small grayscale image to run the robust pipeline. Color (YCbCr 4:2:0) can be added if needed.")
