"""
Primer and barcode design utilities.

We illustrate **combinatorial barcoding**: an image_id barcode * Bi and a layer_id barcode * Lk.
You can increase Hamming distances and add sequence constraints here.
"""
from __future__ import annotations
import itertools
import random

NUC = "ACGT"

def gc_content(seq: str) -> float:
    g = seq.count("G") + seq.count("C")
    return g / max(1, len(seq))

def max_homopolymer(seq: str) -> int:
    m, cur = 1, 1
    for a,b in zip(seq, seq[1:]):
        if a==b:
            cur += 1
            m = max(m, cur)
        else:
            cur = 1
    return m

def hamming(a: str, b: str) -> int:
    return sum(x!=y for x,y in zip(a,b))

def generate_barcode_set(n: int, length: int, gc=(0.4,0.6), max_homo=3, min_hd=3, tries=20000):
    """
    Greedy pick of barcodes satisfying GC, homopolymer, and pairwise Hamming distance.
    """
    cand = []
    rng = random.Random(123)
    for _ in range(tries):
        s = "".join(rng.choice(NUC) for _ in range(length))
        if not (gc[0] <= gc_content(s) <= gc[1]): continue
        if max_homopolymer(s) > max_homo: continue
        if any(hamming(s, t) < min_hd for t in cand): continue
        cand.append(s)
        if len(cand) >= n: break
    if len(cand) < n:
        raise RuntimeError(f"Could not find {n} barcodes; got {len(cand)}. Relax constraints.")
    return cand

class PrimerScheme:
    def __init__(self, n_images:int, n_layers:int, img_len=12, layer_len=10):
        self.img_barcodes = generate_barcode_set(n_images, img_len, min_hd=4)
        self.layer_barcodes = generate_barcode_set(n_layers, layer_len, min_hd=4)

    def primers_for(self, image_id:int, layer_id:int):
        # left/right can be reverse-complemented in a real system; kept simple here
        imgL = self.img_barcodes[image_id]
        imgR = self.img_barcodes[image_id][::-1]
        layL = self.layer_barcodes[layer_id]
        layR = self.layer_barcodes[layer_id][::-1]
        return (imgL, imgR, layL, layR)
