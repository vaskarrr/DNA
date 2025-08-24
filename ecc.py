"""
Toy ECC/UMI utilities. Replace checksum with a real ECC for production.
"""
from __future__ import annotations
import random

NUC = "ACGT"

def umi(length=10, seed=None):
    rng = random.Random(seed)
    return "".join(rng.choice(NUC) for _ in range(length))

def checksum(seq: str) -> str:
    """
    Very toy 2-nt checksum over quaternary values; for demo only.
    """
    m = 0
    for ch in seq:
        m = (m + "ACGT".find(ch)) % 16
    # map 0..15 to two nts (00..11)(00..11)
    table = ["A","C","G","T"]
    hi, lo = divmod(m, 4)
    return table[hi] + table[lo]
