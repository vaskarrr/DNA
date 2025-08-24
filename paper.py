import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import math
import itertools
import base64

#############################
# Utility: 2D DCT / IDCT
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
# Run-Length + Categories (JPEG-like)
#############################

# Category mapping per paper (nts used). We'll map magnitude to these cats.
DNA_CAT_TABLE = [
    (0, 0),
    # cat 1 omitted
    (2, 5),     # cat 2: 1..5
    (3, 25),    # cat 3: 6..25
    (4, 75),    # cat 4: 26..75
    (5, 275),   # cat 5: 76..275
    (6, 775),   # cat 6: 276..775
    (7, 2775),  # cat 7: 776..2775
    (8, 7775),  # cat 8: 2776..7775
]

# Map absolute value to (cat_id, nts_len, min_abs, max_abs)
DNA_CATS = []
# cat_id list: 0,2,3,4,5,6,7,8
cat_ids = [0,2,3,4,5,6,7,8]
prev_max = 0
for cid, max_abs in zip(cat_ids, [0,5,25,75,275,775,2775,7775]):
    if cid == 0:
        DNA_CATS.append((0, 0, 0, 0))
    else:
        nts_len = cid  # by table row order (2..8)
        min_abs = prev_max + 1
        DNA_CATS.append((cid, nts_len, min_abs, max_abs))
    prev_max = max_abs

CAT_BY_ABS = []
for cid, nts_len, lo, hi in DNA_CATS:
    if cid == 0:
        continue
    CAT_BY_ABS.append((lo, hi, cid, nts_len))

# JPEG AC run-length coding constants
EOB = (0, 0)  # End of Block (special symbol)
ZRL = (15, 0) # Zero Run Length (special symbol)

#############################
# Ternary Huffman (for run/category symbols)
#############################

class TernaryNode:
    def __init__(self, sym=None, freq=0, children=None):
        self.sym = sym
        self.freq = freq
        self.children = children or []  # up to 3
    def is_leaf(self):
        return self.sym is not None

def build_ternary_huffman(freqs):
    nodes = [TernaryNode(sym=s, freq=f) for s, f in freqs.items() if f > 0]
    if not nodes:
        return {}, TernaryNode()
    # handle case of 1 or 2 nodes by adding dummies
    while len(nodes) % 2 == 0:  # want 1 mod 2 so we can pick 3 at a time till one remains
        nodes.append(TernaryNode(sym=None, freq=0))
    while len(nodes) > 1:
        nodes.sort(key=lambda n: n.freq)
        a, b, c = nodes[0], nodes[1], nodes[2]
        parent = TernaryNode(sym=None, freq=a.freq + b.freq + c.freq, children=[a,b,c])
        nodes = nodes[3:]
        nodes.append(parent)
    root = nodes[0]
    codes = {}
    def walk(node, prefix):
        if node.is_leaf() and node.sym is not None:
            codes[node.sym] = prefix or [0]  # at least one trit
            return
        for i, child in enumerate(node.children):
            walk(child, prefix + [i])
    walk(root, [])
    return codes, root

#############################
# Goldman mapping trits<->DNA (avoid homopolymers)
#############################

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
        # avoid homopolymer length > 3
        if len(seq) >= 3 and seq[-1] == seq[-2] == seq[-3] == base:
            base = choices[(t+1) % 3]
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
            # if invalid (shouldn't), fall back to 0
            t = 0
        trits.append(t)
        prev = ch
    return trits

#############################
# PAIRCODE-like value coder (deterministic, reversible)
#############################

def value_to_category(v):
    if v == 0:
        return 0, 0, 0, 0
    a = abs(v)
    for lo, hi, cid, nts_len in CAT_BY_ABS:
        if lo <= a <= hi:
            return cid, nts_len, lo, hi
    # clamp to max
    lo, hi, cid, nts_len = CAT_BY_ABS[-1]
    return cid, nts_len, lo, hi

def int_to_trits(n, length):
    # base-3 fixed-length representation
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
    # map v in [ -hi..-lo, 0, lo..hi ] but we never call for 0
    # Create offset index: positives first then negatives (like JPEG value bits sign), deterministic
    # Count of values in category: 2*(hi-lo+1)
    total = 2*(hi - lo + 1)
    if v > 0:
        idx = v - lo
    else:
        idx = (hi - lo + 1) + (abs(v) - lo)
    # represent idx in base-3 using as many trits as possible given nts_len; ensure capacity
    # we target exactly nts_len nucleotides => nts_len trits
    max_vals = 3**nts_len
    if total > max_vals:
        # if overflow, cap (rare for small nts_len vs ranges here); alternatively split across more nts
        idx = min(idx, max_vals-1)
    trits = int_to_trits(idx, nts_len)
    dna = trits_to_dna(trits, start_base=start_base)
    return dna, trits

def decode_value_paircode(dna, nts_len, lo, hi, start_base):
    trits = dna_to_trits(dna[:nts_len], start_base=start_base)
    idx = trits_to_int(trits)
    mid = (hi - lo + 1)
    if idx < mid:
        v = lo + idx
    else:
        v = -(lo + (idx - mid))
    return v, nts_len

#############################
# Encoding pipeline
#############################

def jpeg_like_encode(img_gray, quality=50):
    # img_gray: uint8 0..255
    img = img_gray.astype(np.float32) - 128.0
    blocks, H, W = blockify(img, 8)
    q = quality_scale(quality).astype(np.float32)
    coeffs = []
    for b in blocks:
        d = dct2(b)
        qd = np.round(d / q)
        coeffs.append(qd)
    coeffs = np.array(coeffs)
    # DC/AC streams
    dc_vals = []
    ac_tuples = []  # list of sequences per block
    prev_dc = 0
    for c in coeffs:
        zz = zigzag(c)
        dc = int(zz[0])
        dc_diff = dc - prev_dc
        prev_dc = dc
        dc_vals.append(dc_diff)
        # AC run-length
        ac = zz[1:]
        run = 0
        ac_list = []
        for v in ac:
            v = int(v)
            if v == 0:
                run += 1
                if run == 16:
                    ac_list.append(ZRL)
                    run = 0
            else:
                # determine DNA category for value
                cid, nts_len, lo, hi = value_to_category(v)
                # cat 0 shouldn't happen for nonzero
                ac_list.append((run, cid))
                ac_list.append(('VAL', v))  # value placeholder (encoded later via PAIRCODE)
                run = 0
        if run > 0:
            ac_list.append(EOB)
        else:
            ac_list.append(EOB)
        ac_tuples.append(ac_list)
    return dc_vals, ac_tuples, H, W, q

#############################
# Symbol collection and ternary Huffman for DC categories and AC (run,cat)
#############################

def collect_symbols(dc_vals, ac_tuples):
    # DC: categories for diffs
    dc_syms = []
    for v in dc_vals:
        cid, nts_len, lo, hi = value_to_category(v)
        dc_syms.append(('DC', cid))
    # AC: run/category symbols + EOB/ZRL
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
    from collections import Counter
    dc_freq = Counter(dc_syms)
    ac_freq = Counter(ac_syms)
    dc_codes, dc_root = build_ternary_huffman(dc_freq)
    ac_codes, ac_root = build_ternary_huffman(ac_freq)
    return dc_codes, ac_codes

#############################
# Full DNA encoding (JPEG-DNA style)
#############################

def encode_to_dna(dc_vals, ac_tuples):
    dc_syms, ac_syms = collect_symbols(dc_vals, ac_tuples)
    dc_codes, ac_codes = build_ternary_codes(dc_syms, ac_syms)

    dna_stream = ''
    prev_base = None
    meta = {
        'dc_codes': dc_codes,
        'ac_codes': ac_codes,
    }

    # encode DC
    dc_meta = []
    for v in dc_vals:
        cid, nts_len, lo, hi = value_to_category(v)
        # write DC category (ternary Huffman -> trits -> DNA)
        trits_cat = dc_codes[('DC', cid)]
        dna_cat = trits_to_dna(trits_cat, start_base=prev_base)
        if len(dna_cat) > 0:
            prev_base = dna_cat[-1]
        dna_stream += dna_cat
        # write value via PAIRCODE-like (if nonzero)
        if cid != 0:
            dna_val, _ = encode_value_paircode(v, nts_len, lo, hi, start_base=prev_base)
            if len(dna_val) > 0:
                prev_base = dna_val[-1]
            dna_stream += dna_val
        dc_meta.append({'v': v, 'cid': cid, 'nts_len': nts_len})

    # encode AC
    for L in ac_tuples:
        for item in L:
            if item == EOB:
                trits = ac_codes[('EOB',)]
                dna = trits_to_dna(trits, start_base=prev_base)
                if len(dna) > 0:
                    prev_base = dna[-1]
                dna_stream += dna
            elif item == ZRL:
                trits = ac_codes[('ZRL',)]
                dna = trits_to_dna(trits, start_base=prev_base)
                if len(dna) > 0:
                    prev_base = dna[-1]
                dna_stream += dna
            elif isinstance(item, tuple) and item[0] != 'VAL':
                run, cid = item
                trits = ac_codes[('AC', run, cid)]
                dna = trits_to_dna(trits, start_base=prev_base)
                if len(dna) > 0:
                    prev_base = dna[-1]
                dna_stream += dna
            elif isinstance(item, tuple) and item[0] == 'VAL':
                v = item[1]
                cid, nts_len, lo, hi = value_to_category(v)
                if cid != 0:
                    dna_val, _ = encode_value_paircode(v, nts_len, lo, hi, start_base=prev_base)
                    if len(dna_val) > 0:
                        prev_base = dna_val[-1]
                    dna_stream += dna_val
    meta['dc_vals'] = dc_vals
    meta['H'] = None
    return dna_stream, meta

#############################
# Decoding (lossless in our stream; lossy only from JPEG quantization)
#############################

# For decoding we need to traverse ternary Huffman trees. Rebuild from code tables.

def rebuild_tree_from_codes(codes):
    root = TernaryNode()
    for sym, trits in codes.items():
        node = root
        for t in trits:
            while len(node.children) <= t:
                node.children.append(TernaryNode())
            if node.children[t] is None:
                node.children[t] = TernaryNode()
            node = node.children[t]
        node.sym = sym
    return root

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

def decode_from_dna(dna_stream, dc_codes, ac_codes, blocks_count):
    dc_root = rebuild_tree_from_codes(dc_codes)
    ac_root = rebuild_tree_from_codes(ac_codes)
    pos = 0
    prev_base = None
    dc_vals = []
    # decode DC categories and values
    for _ in range(blocks_count):
        sym, pos, prev_base = read_symbol_from_dna(dna_stream, pos, dc_root, prev_base)
        assert sym[0] == 'DC'
        cid = sym[1]
        if cid == 0:
            v = 0
        else:
            # read nts_len nucleotides and map back
            nts_len = cid
            v, _ = decode_value_paircode(dna_stream[pos:pos+nts_len], nts_len, *[s for s in next(( (lo,hi) for lo,hi,c,l in CAT_BY_ABS if c==cid ), (1,1))], start_base=prev_base)
            pos += nts_len
            prev_base = dna_stream[pos-1]
        dc_vals.append(v)
    # decode AC for each block until EOB
    ac_lists = []
    for _ in range(blocks_count):
        L = []
        while True:
            sym, pos, prev_base = read_symbol_from_dna(dna_stream, pos, ac_root, prev_base)
            if sym == ('EOB',):
                L.append(EOB)
                break
            elif sym == ('ZRL',):
                L.append(ZRL)
            elif sym[0] == 'AC':
                run, cid = sym[1], sym[2]
                L.append((run, cid))
                if cid != 0:
                    nts_len = cid
                    lo, hi = next(((lo,hi) for lo,hi,c,l in CAT_BY_ABS if c==cid), (1,1))
                    v, _ = decode_value_paircode(dna_stream[pos:pos+nts_len], nts_len, lo, hi, start_base=prev_base)
                    pos += nts_len
                    prev_base = dna_stream[pos-1]
                    L.append(('VAL', v))
        ac_lists.append(L)
    return dc_vals, ac_lists

#############################
# Reconstruct image from DC/AC streams
#############################

def rebuild_blocks(dc_vals, ac_lists, H, W, q):
    blocks_n = len(dc_vals)
    blocks = []
    prev_dc = 0
    for i in range(blocks_n):
        dc_diff = dc_vals[i]
        dc = prev_dc + dc_diff
        prev_dc = dc
        # build AC coefficients
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
                # next is VAL tuple
                j += 1
                val = 0
                if j < len(ac) and isinstance(ac[j], tuple) and ac[j][0] == 'VAL':
                    val = ac[j][1]
                coef[k] = val
                k += 1
            j += 1
        block = izigzag(coef).reshape(8,8)
        blocks.append(block)
    blocks = np.array(blocks)
    # de-quantize and IDCT
    qf = q.astype(np.float32)
    out_blocks = []
    for b in blocks:
        deq = b * qf
        img_b = idct2(deq)
        out_blocks.append(img_b)
    out = deblockify(np.array(out_blocks), H, W, 8)
    out = np.clip(out + 128.0, 0, 255).astype(np.uint8)
    return out

#############################
# DNA Utilities: GC content, homopolymer, oligo split
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

st.set_page_config(page_title="JPEG-DNA Prototype", layout="wide")
st.title("🧬 JPEG-DNA Image Coding (Prototype)")
st.caption("Didactic prototype inspired by JPEG-DNA with ternary Huffman (Goldman) + PAIRCODE-like value coding. Avoids long homopolymers and shows GC balance.\nThis is a research demo, not a biology-grade production encoder.")

with st.sidebar:
    st.header("Parameters")
    quality = st.slider("JPEG Quality", 10, 95, 60, step=1)
    oligo_len = st.slider("Oligo length (nt)", 80, 300, 220, step=10)
    simulate_error = st.checkbox("Simulate 1 random deletion in DNA stream", value=False)

uploaded = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg","jpeg","png"]) 

col1, col2 = st.columns([1,1])

if uploaded:
    img = Image.open(uploaded).convert('L')
    img_np = np.array(img)
    col1.image(img, caption=f"Original ({img_np.shape[1]}×{img_np.shape[0]})", use_column_width=True)

    # Encode
    dc_vals, ac_lists, H, W, q = jpeg_like_encode(img_np, quality)
    blocks_count = len(dc_vals)

    dna_stream, meta = encode_to_dna(dc_vals, ac_lists)

    # Optional error simulation
    if simulate_error and len(dna_stream) > 0:
        import random
        pos = random.randrange(len(dna_stream))
        dna_stream = dna_stream[:pos] + dna_stream[pos+1:]

    # Split into oligos
    oligos = split_into_oligos(dna_stream, oligo_len=oligo_len)

    # Stats
    total_nt = len(dna_stream)
    gc = gc_content(dna_stream)
    hp = max_homopolymer(dna_stream)

    with col2:
        st.subheader("DNA Stream Stats")
        st.metric("Total nucleotides", f"{total_nt:,}")
        st.metric("GC content (%)", f"{gc:.2f}")
        st.metric("Max homopolymer length", f"{hp}")
        st.write(f"Oligos: {len(oligos)} (length≈{oligo_len} nt)")
        st.code(dna_stream[:300] + ("..." if len(dna_stream) > 300 else ""), language="text")

    st.divider()
    st.subheader("Reconstruction (decoder)")
    try:
        dc_codes = meta['dc_codes']
        ac_codes = meta['ac_codes']
        # decode
        d_dc, d_ac = decode_from_dna(''.join(seq for _, seq in oligos), dc_codes, ac_codes, blocks_count)
        recon = rebuild_blocks(d_dc, d_ac, H, W, q)
        # Crop to original size
        recon = recon[:img_np.shape[0], :img_np.shape[1]]
        st.image(recon, caption="Decoded image (from DNA)", use_column_width=False)
        # PSNR
        mse = np.mean((recon.astype(np.float32) - img_np.astype(np.float32))**2)
        psnr = 10*np.log10(255*255/(mse+1e-9))
        st.write(f"PSNR vs original (due to quantization): **{psnr:.2f} dB**")
    except Exception as e:
        st.warning(f"Decoding failed (likely due to variable-length shift from the simulated deletion or symbol ambiguity): {e}")

    # Downloads
    st.subheader("Downloads")
    # FASTA
    fasta = []
    for name, seq in oligos:
        fasta.append(f">{name}\n{seq}")
    fasta_txt = "\n".join(fasta)
    st.download_button("Download oligos (FASTA)", data=fasta_txt, file_name="oligos.fasta", mime="text/plain")

    # Raw DNA
    st.download_button("Download raw DNA stream (TXT)", data=dna_stream, file_name="dna_stream.txt", mime="text/plain")

    # Metadata (brief)
    meta_txt = {
        'blocks': blocks_count,
        'quality': quality,
        'oligo_len': oligo_len,
        'dc_code_symbols': len(meta['dc_codes']),
        'ac_code_symbols': len(meta['ac_codes']),
        'gc_percent': gc,
        'max_homopolymer': hp,
    }
    st.json(meta_txt)

else:
    st.info("Upload a small grayscale image to try the pipeline. For color images, this demo uses the luminance channel only.")
