# -*- coding: utf-8 -*-
"""Export UCI .mat datasets to a compact binary format for the C++ solver.

Format (unified, both dense and sparse):
  <int32 m> <int32 n> <int32 sparse_flag> <int64 nnz>
  sparse_flag == 0 (dense): A as m*n doubles row-major, then b as m doubles
  sparse_flag == 1 (CSC) : colptr[n+1] int32, rowidx[nnz] int32,
                           vals[nnz] double, then b as m doubles
Also writes <name>.json with {name, m, n, nnz, sparse, norm_b, source}.

Usage: python tests/export_ucidata.py [name ...]   (default: all .mat)
"""
import json
import os
import struct
import sys

import numpy as np
import scipy.io
import scipy.sparse

DATA_DIR = r"C:\Users\qianl\OneDrive\codes\Group lasso\UCIdata"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ucidata")


def export_one(mat_file):
    src = os.path.join(DATA_DIR, mat_file)
    d = scipy.io.loadmat(src)
    key = "A" if "A" in d else [k for k in d if not k.startswith("__")][0]
    A = d[key]
    bkey = "b" if "b" in d else None
    if bkey is None:
        others = [k for k in d if not k.startswith("__") and k != key]
        bkey = others[0] if others else None
    if bkey is None:
        raise RuntimeError(f"no response variable in {mat_file}")
    b = np.asarray(d[bkey], dtype=np.float64).ravel()
    m = b.size

    if scipy.sparse.issparse(A):
        A = A.tocsc().astype(np.float64)
        if A.shape[0] != m:
            A = A.T.tocsc()
        n = A.shape[1]
        nnz = A.nnz
        sparse = True
    else:
        A = np.asarray(A, dtype=np.float64)
        n = A.size // m
        if A.size != m * n:
            raise RuntimeError(f"shape mismatch: A={A.shape} b={b.shape}")
        if A.shape[0] != m:
            A = A.T
        A = np.ascontiguousarray(A.reshape(m, n))
        nnz = 0
        sparse = False

    name = os.path.splitext(mat_file)[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    bin_path = os.path.join(OUT_DIR, name + ".bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<iiiq", m, n, 1 if sparse else 0, int(nnz)))
        if sparse:
            f.write(np.asarray(A.indptr, dtype="<i4").tobytes())      # n+1
            f.write(np.asarray(A.indices, dtype="<i4").tobytes())     # nnz
            f.write(A.data.astype("<f8").tobytes())
        else:
            f.write(A.astype("<f8").tobytes())
        f.write(b.astype("<f8").tobytes())

    meta = dict(name=name, m=int(m), n=int(n), nnz=int(nnz), sparse=sparse,
                norm_b=float(np.linalg.norm(b)), source=mat_file,
                note="UCI dataset exported for the C++ BMOP test")
    with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{name}: m={m} n={n} sparse={sparse} nnz={nnz} "
          f"norm_b={meta['norm_b']:.4f} bin={os.path.getsize(bin_path)/1e6:.1f} MB")


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".mat"))
    for t in targets:
        try:
            export_one(t)
        except Exception as e:
            print(f"SKIP {t}: {e!r}")


if __name__ == "__main__":
    main()
