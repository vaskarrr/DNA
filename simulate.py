"""
CLI simulation driver.
"""
from __future__ import annotations
import os, argparse, glob
from PIL import Image
import numpy as np
from .primers import PrimerScheme
from .core import progressive_layers, make_oligos_for_layer, write_pool_fasta, reconstruct_from_oligos, mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with JPG/PNG images")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--block_nt", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(args.images, "*.*")))
    primers = PrimerScheme(n_images=len(imgs), n_layers=args.layers)

    report = []

    for img_id, path in enumerate(imgs):
        img = Image.open(path).convert("RGB")
        layers = progressive_layers(img, n_layers=args.layers)

        for layer_id, layer_img in enumerate(layers):
            arr = np.array(layer_img)
            oligos = make_oligos_for_layer(img_id, layer_id, arr, primers, data_block_nt=args.block_nt)
            pool_path = os.path.join(args.out, f"image{img_id}_layer{layer_id}.fa")
            write_pool_fasta(oligos, pool_path)

        # quick round-trip check: reconstruct highest layer
        full_arr = np.array(layers[-1])
        # (we skip decoding PNG here; in a real pipeline we'd read bytes back to image)
        # Just compute proxy distortion vs Lk upscaled for illustration
        for k, layer_img in enumerate(layers):
            up = layer_img.resize(img.size, Image.BICUBIC)
            d = mse(np.array(up), full_arr)
            report.append(dict(image=path, layer=k, mse=float(d)))

    # write a small JSON with distortion proxies
    with open(os.path.join(args.out, "report.json"), "w") as f:
        import json; json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
