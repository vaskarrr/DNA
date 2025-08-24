"""
Core encode/decode and simulation helpers.
"""
from __future__ import annotations
import os, io, math, random, json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np

from .primers import PrimerScheme
from .ecc import umi, checksum

MAP2DNA = {"00":"A","01":"C","10":"G","11":"T"}
DNA2MAP = {v:k for k,v in MAP2DNA.items()}

def bytes_to_dna(b: bytes) -> str:
    bits = "".join(f"{x:08b}" for x in b)
    # pad to multiple of 2
    if len(bits) % 2: bits += "0"
    return "".join(MAP2DNA[bits[i:i+2]] for i in range(0, len(bits), 2))

def dna_to_bytes(seq: str) -> bytes:
    bits = "".join(DNA2MAP[c] for c in seq)
    # trim to multiple of 8
    bits = bits[:len(bits)//8*8]
    return int(bits, 2).to_bytes(len(bits)//8, "big") if bits else b""

def constrain_seq(seq: str, max_homo=3, gc=(0.4,0.6)):
    # Simple rejection sampler: in real life use balanced coders / constrained coding.
    def ok(s):
        g = (s.count("G")+s.count("C"))/max(1,len(s))
        # homopolymer
        m, cur = 1,1
        for a,b in zip(s,s[1:]):
            if a==b:
                cur += 1; m=max(m,cur)
            else:
                cur = 1
        return (gc[0] <= g <= gc[1]) and (m <= max_homo)
    if ok(seq): return seq
    # naive shuffle to try improve constraints
    s = list(seq)
    random.Random(42).shuffle(s)
    s = "".join(s)
    return s if ok(s) else seq  # fallback

@dataclass
class Oligo:
    imgL: str; imgR: str; layL: str; layR: str
    umi: str; payload: str; csum: str

    def sequence(self) -> str:
        return f"{self.layL}{self.imgL}{self.umi}{self.payload}{self.csum}{self.imgR}{self.layR}"

def progressive_layers(img: Image.Image, n_layers=5) -> List[Image.Image]:
    """
    L0 is thumbnail (small), L{n-1} is full res.
    Each layer doubles width/height.
    """
    base_w, base_h = img.size
    layers = []
    for k in range(n_layers):
        scale = 2**(k - (n_layers-1))  # L0 = 1/2^(n-1)
        w = max(1, int(base_w * max(scale, 1/base_w)))
        h = max(1, int(base_h * max(scale, 1/base_h)))
        lyr = img.resize((w,h), Image.BICUBIC)
        layers.append(lyr)
    return layers

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.mean((a-b)**2))

def make_oligos_for_layer(image_id:int, layer_id:int, arr: np.ndarray, primers: PrimerScheme, data_block_nt=200):
    # serialize image as PNG bytes
    im = Image.fromarray(arr)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    data = buf.getvalue()
    dna = bytes_to_dna(data)
    dna = constrain_seq(dna)
    imgL, imgR, layL, layR = primers.primers_for(image_id, layer_id)
    # chunk into payload blocks
    oligos = []
    for i in range(0, len(dna), data_block_nt):
        payload = dna[i:i+data_block_nt]
        tag = umi(10, seed=(image_id*1315423911 + layer_id*2654435761 + i))
        csum = checksum(payload)
        oligos.append(Oligo(imgL, imgR, layL, layR, tag, payload, csum))
    return oligos

def error_channel(seq: str, p_sub=0.002, p_ins=0.001, p_del=0.001, seed=123):
    rng = random.Random(seed)
    out = []
    for ch in seq:
        r = rng.random()
        if r < p_del:
            continue
        elif r < p_del + p_ins:
            out.append(rng.choice("ACGT"))
            out.append(ch)
        elif r < p_del + p_ins + p_sub:
            out.append(rng.choice([x for x in "ACGT" if x!=ch]))
        else:
            out.append(ch)
    return "".join(out)

def naive_consensus(reads: List[str]) -> str:
    if not reads: return ""
    # align by length (toy). Real pipeline should do POA/MSA.
    L = min(len(r) for r in reads)
    reads = [r[:L] for r in reads]
    cols = list(zip(*reads))
    cons = []
    for col in cols:
        # majority vote
        base = max(set(col), key=col.count)
        cons.append(base)
    return "".join(cons)

def reconstruct_from_oligos(oligos: List[Oligo]) -> bytes:
    # sort by UMI order (as proxy for original order); in real design include explicit indices.
    oligos_sorted = sorted(oligos, key=lambda o:o.umi)
    payload = "".join(o.payload for o in oligos_sorted)
    return dna_to_bytes(payload)

def write_pool_fasta(oligos: List[Oligo], path:str):
    with open(path, "w") as f:
        for i, o in enumerate(oligos):
            f.write(f">oligo_{i}\n{o.sequence()}\n")

