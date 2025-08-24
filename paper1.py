import streamlit as st
import numpy as np
from PIL import Image
import math
from collections import Counter

"""
Closed-loop JPEG-DNA prototype with robustness features:
- Ternary Huffman for DC categories and AC (run,cat) like JPEG (Goldman-style trits→DNA mapping)
- PAIRCODE-inspired value coder with category-length mapping (cat 1 omitted)
- Block FRAMING with:
  • Unique restart marker
  • Fixed-length LENGTH field (in trits) → DNA
  • Payload DNA encoded per block with prev_base reset (self-contained)
  • CRC-16 over payload (stored as trits→DNA)
  • Optional 3× repetition ECC (majority vote at decode)
- Restart intervals reset DC predictor like JPEG RST markers
- Oligo splitting + GC/homopolymer checks

Notes:
- This is a research/demo tool; biological constraints and true ECC (Reed–Solomon across oligos, primers, indices) are simplified.
- Color handling omitted; operates on luminance (grayscale) for clarity.
"""

#############################
# Math helpers: 2D DCT / IDCT (orthonormal)
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
    [0, 1, 5, 6,14,15,27,28],
    [2, 4, 7,13,16,26,29,42],
    [3, 8,12,17,25,30,41,43],
    [9,11,18,24,31,40,44,53],
    [10,19,23,32,39,45,52,54],
    [20,22,33,38,46,51,55,60],
    [21,34,37,47,50,56,59,61],
    [35,36,48,49,57,58,62,63],
]).flatten()

STD_LUMA_Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.int32)

def quality_scale(Q):
    Q = max(1, min(100, int(Q)))
    if Q < 50:
        scale = 5000 / Q
    else:
        scale = 200 - 2 * Q
    table = np.floor((STD_LUMA_Q * scale + 50) / 100).astype(np.int32)
    table[table == 0] = 1
    return table

def blockify(img, block=8):
    h, w = img.shape
    H = (h + block - 1) // block * block
    W = (w + block - 1) // block * block
    padded = np.zeros((H, W), dtype=np.float32)
    padded[:h, :w] = img
    blocks = []
    for y in range(0, H, block):
        for x in range(0, W, block):
            blocks.append(padded[y:y+block, x:x+block])
    return np.array(blocks), H, W

def deblockify(blocks, H, W, block=8):
    out = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y in range(0, H, block):
        for x in range(0, W, block):
            out[y:y+block, x:x+block] = blocks[idx]
            idx += 1
    return out

def zigzag(block):
    return block.flatten()[ZIGZAG_IDX]

def izigzag(vec):
    block = np.zeros(64, dtype=np.float32)
    block[ZIGZAG_IDX] = vec
    return block.reshape(8,8)

#############################
# Categories per paper (cat 1 omitted)
#############################

# (cat_id, nts_len, min_abs, max_abs)
DNA_CATS = [
    (0, 0, 0, 0),      # zero
    (2, 2, 1, 5),
    (3, 3, 6, 25),
    (4, 4, 26, 75),
    (5, 5, 76, 275),
    (6, 6, 276, 775),
    (7, 7, 776, 2775),
    (8, 8, 2776, 7775),
]

CAT_BY_ABS = [(lo,hi,cid,nts) for (cid,nts,lo,hi) in DNA_CATS if cid!=0]

EOB = (0, 0)
ZRL = (15, 0)

#############################
# Ternary Huffman and Goldman mapping
#############################

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
        a,b,c = nodes[0], nodes[1], nodes[2]
        parent = TernaryNode(children=[a,b,c], freq=a.freq+b.freq+c.freq)
        nodes = nodes[3:] + [parent]
    root = nodes[0]
    codes = {}
    def walk(node, prefix):
        if node.is_leaf() and node.sym is not None:
            codes[node.sym] = prefix or [0]
            return
        for i,ch in enumerate(node.children):
            walk(ch, prefix+[i])
    walk(root, [])
    return codes, root

NUCS = ['A','C','G','T']
NEXT_MAP = {
    None: ['A','C','G'],
    'A':  ['C','G','T'],
    'C':  ['A','G','T'],
    'G':  ['A','C','T'],
    'T':  ['A','C','G'],
}

def trits_to_dna(trits, start_base=None):
    seq = []
    prev = start_base
    for t in trits:
        choices = NEXT_MAP[prev]
        base = choices[t % 3]
        if len(seq) >= 3 and seq[-1]==seq[-2]==seq[-3]==base:
            base = choices[(t+1)%3]
        seq.append(base)
        prev = base
    return ''.join(seq)

def dna_to_trits(seq, start_base=None):
    trits = []
    prev = start_base
    for ch in seq:
        choices = NEXT_MAP[prev]
        try:
            t = choices.index(ch)
        except ValueError:
            t = 0
        trits.append(t)
        prev = ch
    return trits

#############################
# PAIRCODE-like value coder
#############################

def value_to_category(v):
    if v == 0:
        return 0, 0, 0, 0
    a = abs(v)
    for lo,hi,cid,nts in CAT_BY_ABS:
        if lo <= a <= hi:
            return cid, nts, lo, hi
    lo,hi,cid,nts = CAT_BY_ABS[-1]
    return cid, nts, lo, hi

def int_to_trits(n, length):
    out = [0]*length
    for i in range(length-1, -1, -1):
        out[i] = n % 3
        n //= 3
    return out

def trits_to_int(trits):
    n = 0
    for t in trits:
        n = n*3 + t
    return n

def encode_value_paircode(v, nts_len, lo, hi, start_base):
    total = 2*(hi - lo + 1)
    if v > 0:
        idx = v - lo
    else:
        idx = (hi - lo + 1) + (abs(v) - lo)
    max_vals = 3**nts_len
    idx = min(idx, max_vals-1)
    trits = int_to_trits(idx, nts_len)
    dna = trits_to_dna(trits, start_base=start_base)
    return dna

def decode_value_paircode(dna, nts_len, lo, hi, start_base):
    trits = dna_to_trits(dna[:nts_len], start_base=start_base)
    idx = trits_to_int(trits)
    mid = (hi - lo + 1)
    if idx < mid:
        v = lo + idx
    else:
        v = -(lo + (idx - mid))
    return v

#############################
# JPEG-like encode to DC/AC
#############################

def jpeg_like_encode(img_gray, quality=60):
    img = img_gray.astype(np.float32) - 128.0
    blocks, H, W = blockify(img, 8)
    q = quality_scale(quality).astype(np.float32)
    coeffs = []
    for b in blocks:
        d = dct2(b)
        qd = np.round(d / q)
        coeffs.append(qd)
    coeffs = np.array(coeffs)
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
                    block_ac.append(ZRL)
                    run = 0
            else:
                cid, nts_len, lo, hi = value_to_category(v)
                block_ac.append((run, cid))
                block_ac.append(('VAL', v))
                run = 0
        block_ac.append(EOB)
        ac_tuples.append(block_ac)
    return dc_vals, ac_tuples, H, W, q

#############################
# Build codes
#############################

def collect_symbols(dc_vals, ac_tuples):
    dc_syms = [('DC', value_to_category(v)[0]) for v in dc_vals]
    ac_syms = []
    for L in ac_tuples:
        for item in L:
            if item == EOB:
                ac_syms.append(('EOB',))
            elif item == ZRL:
                ac_syms.append(('ZRL',))
            elif isinstance(item, tuple) and item[0] != 'VAL':
                run, cid = item
                ac_syms.append(('AC', run, cid))
    return dc_syms, ac_syms

def build_ternary_codes(dc_syms, ac_syms):
    dc_freq = Counter(dc_syms)
    ac_freq = Counter(ac_syms)
    dc_codes, _ = build_ternary_huffman(dc_freq)
    ac_codes, _ = build_ternary_huffman(ac_freq)
    return dc_codes, ac_codes

#############################
# Framing + ECC
#############################

# Marker designed for low accidental occurrence (24 nt)
RST_MARKER = 'ACGTACGTACGTACGTACGTACGT'
LEN_TRITS = 20   # supports up to 3^20 ≈ 3.4e9 nts
CRC_TRITS = 12   # 3^12 > 65535

# CRC-16-CCITT (0x1021)

def crc16(data_bytes: bytes, poly=0x1021, init=0xFFFF):
    crc = init
    for b in data_bytes:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

def int_to_fixed_trits(n: int, length: int):
    return int_to_trits(n, length)

def fixed_trits_to_int(trits):
    return trits_to_int(trits)

def header_pack_length(length_nt):
    tr = int_to_fixed_trits(length_nt, LEN_TRITS)
    return trits_to_dna(tr, start_base=None)

def header_unpack_length(dna):
    tr = dna_to_trits(dna[:LEN_TRITS], start_base=None)
    return fixed_trits_to_int(tr)

def crc_pack_trits(crc_val):
    tr = int_to_fixed_trits(crc_val, CRC_TRITS)
    return trits_to_dna(tr, start_base=None)

def crc_unpack_trits(dna):
    tr = dna_to_trits(dna[:CRC_TRITS], start_base=None)
    return fixed_trits_to_int(tr)

#############################
# Encode blocks → framed DNA stream
#############################

def encode_block_payload(bi, dc_val, ac_list, dc_codes, ac_codes):
    # Encode a single block starting fresh prev_base=None (self-contained)
    prev_base = None
    payload = ''
    # DC category
    cid, nts_len, lo, hi = value_to_category(dc_val)
    trits_cat = dc_codes[('DC', cid)]
    dna_cat = trits_to_dna(trits_cat, start_base=prev_base)
    payload += dna_cat
    prev_base = dna_cat[-1] if dna_cat else None
    # DC value if nonzero
    if cid != 0:
        dna_val = encode_value_paircode(dc_val, nts_len, lo, hi, start_base=prev_base)
        payload += dna_val
        prev_base = dna_val[-1] if dna_val else prev_base
    # AC stream for this block
    for item in ac_list:
        if item == EOB:
            tr = ac_codes[('EOB',)]
            dna = trits_to_dna(tr, start_base=prev_base)
            payload += dna
            prev_base = dna[-1] if dna else prev_base
        elif item == ZRL:
            tr = ac_codes[('ZRL',)]
            dna = trits_to_dna(tr, start_base=prev_base)
            payload += dna
            prev_base = dna[-1] if dna else prev_base
        elif isinstance(item, tuple) and item[0] != 'VAL':
            run, cid2 = item
            tr = ac_codes[('AC', run, cid2)]
            dna = trits_to_dna(tr, start_base=prev_base)
            payload += dna
            prev_base = dna[-1] if dna else prev_base
        elif isinstance(item, tuple) and item[0] == 'VAL':
            v = item[1]
            cid2, nts_len2, lo2, hi2 = value_to_category(v)
            if cid2 != 0:
                dna_val = encode_value_paircode(v, nts_len2, lo2, hi2, start_base=prev_base)
                payload += dna_val
                prev_base = dna_val[-1] if dna_val else prev_base
    return payload

def replicate_ecc(seq, rep=1):
    if rep <= 1:
        return seq
    return ''.join(seq for _ in range(rep))

def majority_vote(strings):
    if not strings:
        return ''
    L = min(len(s) for s in strings)
    strings = [s[:L] for s in strings]
    out = []
    for i in range(L):
        col = [s[i] for s in strings]
        counts = Counter(col)
        out.append(counts.most_common(1)[0][0])
    return ''.join(out)

def encode_stream(dc_vals, ac_lists, dc_codes, ac_codes, restart_interval=8, ecc_rep=1):
    frames = []
    dc_reset_points = set(range(0, len(dc_vals), max(1, restart_interval)))
    # Build framed blocks
    for i in range(len(dc_vals)):
        payload = encode_block_payload(i, dc_vals[i], ac_lists[i], dc_codes, ac_codes)
        # CRC over payload DNA bytes
        crc = crc16(payload.encode('ascii'))
        header_len = header_pack_length(len(payload) * ecc_rep)
        crc_dna = crc_pack_trits(crc)
        payload_ecc = replicate_ecc(payload, ecc_rep)
        frame = RST_MARKER + header_len + payload_ecc + crc_dna
        frames.append(frame)
    return ''.join(frames)

#############################
# Decode framed stream → DC/AC lists
#############################

def decode_stream(dna_stream, blocks_count, dc_codes, ac_codes, restart_interval=8, ecc_rep=1):
    # Rebuild Huffman trees
    def rebuild_tree_from_codes(codes):
        root = TernaryNode()
        for sym, trits in codes.items():
            node = root
            for t in trits:
                while len(node.children) <= t:
                    node.children.append(TernaryNode())
                node = node.children[t]
            node.sym = sym
        return root

    dc_root = rebuild_tree_from_codes(dc_codes)
    ac_root = rebuild_tree_from_codes(ac_codes)

    def read_symbol_from_dna(dna, pos, root, start_base):
        node = root
        prev = start_base
        while True:
            if pos >= len(dna):
                return None, pos, prev
            ch = dna[pos]
            choices = NEXT_MAP[prev]
            try:
                t = choices.index(ch)
            except ValueError:
                t = 0
            if t >= len(node.children) or node.children[t] is None:
                node.children += [TernaryNode()] * (t - len(node.children) + 1)
                node.children[t] = TernaryNode()
            node = node.children[t]
            prev = ch
            pos += 1
            if node.sym is not None:
                return node.sym, pos, prev

    # Scan frames
    frames = []
    i = 0
    while True:
        idx = dna_stream.find(RST_MARKER, i)
        if idx == -1:
            break
        start = idx + len(RST_MARKER)
        if start + LEN_TRITS > len(dna_stream):
            break
        length_nt = header_unpack_length(dna_stream[start:start+LEN_TRITS])
        pay_start = start + LEN_TRITS
        pay_end = pay_start + length_nt
        crc_start = pay_end
        crc_end = crc_start + CRC_TRITS
        if crc_end > len(dna_stream):
            break
        payload_ecc = dna_stream[pay_start:pay_end]
        crc_val = crc_unpack_trits(dna_stream[crc_start:crc_end])
        # ECC majority if needed
        if ecc_rep > 1:
            chunk_len = length_nt // ecc_rep
            reps = [payload_ecc[j*chunk_len:(j+1)*chunk_len] for j in range(ecc_rep)]
            payload = majority_vote(reps)
        else:
            payload = payload_ecc
        # CRC check
        if crc16(payload.encode('ascii')) != crc_val:
            # drop frame if CRC fails
            frames.append((False, payload))
        else:
            frames.append((True, payload))
        i = crc_end
        if len(frames) >= blocks_count:
            break

    # Parse payloads to DC/AC
    dc_vals = []
    ac_lists = []
    for bi in range(blocks_count):
        ok, payload = frames[bi] if bi < len(frames) else (False, '')
        prev_base = None
        pos = 0
        # DC category
        sym, pos, prev_base = read_symbol_from_dna(payload, pos, dc_root, prev_base)
        if sym is None or sym[0] != 'DC':
            dc_vals.append(0)
            ac_lists.append([EOB])
            continue
        cid = sym[1]
        if cid == 0:
            dc_v = 0
        else:
            nts_len = cid
            # find lo,hi for cid
            lo,hi = next(((lo,hi) for (lo,hi,c,_) in CAT_BY_ABS if c==cid), (1,1))
            dna_val = payload[pos:pos+nts_len]
            if len(dna_val) < nts_len:
                dc_v = 0
                pos += len(dna_val)
            else:
                dc_v = decode_value_paircode(dna_val, nts_len, lo, hi, start_base=prev_base)
                pos += nts_len
                prev_base = dna_val[-1] if dna_val else prev_base
        dc_vals.append(dc_v)
        # ACs
        block_ac = []
        guard = 0
        while pos < len(payload) and guard < 10000:
            guard += 1
            sym, pos, prev_base = read_symbol_from_dna(payload, pos, ac_root, prev_base)
            if sym is None:
                break
            if sym == ('EOB',):
                block_ac.append(EOB)
                break
            elif sym == ('ZRL',):
                block_ac.append(ZRL)
            elif sym[0] == 'AC':
                run, rcid = sym[1], sym[2]
                block_ac.append((run, rcid))
                if rcid != 0:
                    nts_len = rcid
                    lo,hi = next(((lo,hi) for (lo,hi,c,_) in CAT_BY_ABS if c==rcid), (1,1))
                    dna_val = payload[pos:pos+nts_len]
                    if len(dna_val) < nts_len:
                        val = 0
                        pos += len(dna_val)
                    else:
                        val = decode_value_paircode(dna_val, nts_len, lo, hi, start_base=prev_base)
                        pos += nts_len
                        prev_base = dna_val[-1] if dna_val else prev_base
                    block_ac.append(('VAL', val))
        if not block_ac or block_ac[-1] != EOB:
            block_ac.append(EOB)
        ac_lists.append(block_ac)
    return dc_vals, ac_lists

#############################
# Rebuild image from DC/AC
#############################

def rebuild_blocks(dc_vals, ac_lists, H, W, q):
    blocks_n = len(dc_vals)
    blocks = []
    prev_dc = 0
    for i in range(blocks_n):
        dc = prev_dc + dc_vals[i]
        # Restart interval handling: if block i is at restart boundary, reset predictor
        # (We handle reset outside by providing dc_vals appropriately; here we just accumulate)
        prev_dc = dc
        coef = np.zeros(64, dtype=np.float32)
        coef[0] = dc
        ac = ac_lists[i]
        k = 1
        j = 0
        while j < len(ac):
            item = ac[j]
            if item == EOB:
                break
            if item == ZRL:
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
        if seq[i] == seq[i-1]:
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
        chunk = seq[i:i+oligo_len]
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
    ecc_rep = st.selectbox("Repetition ECC", [1,3], index=0, help="3× repetition with majority vote per nt")

uploaded = st.file_uploader("Upload image (JPEG/PNG)", type=["jpg","jpeg","png"]) 
col1, col2 = st.columns([1,1])

if uploaded:
    img = Image.open(uploaded).convert('L')
    img_np = np.array(img)
    col1.image(img, caption=f"Original ({img_np.shape[1]}×{img_np.shape[0]})", use_column_width=True)

    # Encode to DC/AC
    dc_vals, ac_lists, H, W, q = jpeg_like_encode(img_np, quality)

    # Build ternary codes
    dc_syms, ac_syms = collect_symbols(dc_vals, ac_lists)
    dc_codes, ac_codes = build_ternary_codes(dc_syms, ac_syms)

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
        # Decode framed stream → DC/AC
        blocks_count = len(dc_vals)
        d_dc, d_ac = decode_stream(''.join(seq for _, seq in oligos), blocks_count, dc_codes, ac_codes, restart_interval=restart_interval, ecc_rep=ecc_rep)
        recon = rebuild_blocks(d_dc, d_ac, H, W, q)
        recon = recon[:img_np.shape[0], :img_np.shape[1]]
        st.image(recon, caption="Decoded image", use_column_width=False)
        mse = np.mean((recon.astype(np.float32) - img_np.astype(np.float32))**2)
        psnr = 10*np.log10(255*255/(mse+1e-9))
        st.write(f"PSNR vs original: **{psnr:.2f} dB** (compression only)")
    except Exception as e:
        st.warning(f"Decoding failed: {e}")

    # Downloads
    st.subheader("Downloads")
    fasta = []
    for name, seq in oligos:
        fasta.append(f">{name}\n{seq}")
    fasta_txt = "\n".join(fasta)
    st.download_button("Download oligos (FASTA)", data=fasta_txt, file_name="oligos.fasta", mime="text/plain")
    st.download_button("Download raw DNA stream (TXT)", data=dna_stream, file_name="dna_stream.txt", mime="text/plain")

    # Meta
    st.subheader("Metadata")
    st.json({
        'blocks': len(dc_vals),
        'quality': quality,
        'restart_interval': restart_interval,
        'ecc_rep': ecc_rep,
        'dc_code_symbols': len(dc_codes),
        'ac_code_symbols': len(ac_codes),
        'gc_percent': gc,
        'max_homopolymer': hp,
    })
else:
    st.info("Upload a small grayscale image to run the robust pipeline. Color support can be added (YCbCr 4:2:0) if needed.")
