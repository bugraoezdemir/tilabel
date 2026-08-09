"""Synthetic correctness tests for scalabel.legacy.label_array_legacy.

Mask and output are backed by real on-disk zarr arrays (not plain numpy or
in-memory stores), since the read-modify-write path in `apply_tile` (Pass 2)
behaves differently for zarr (getitem returns a copy) than for in-memory
numpy (getitem returns a view). Fixtures live under a local _fixtures/
directory (sibling of this tests/ directory, within the package).
"""

from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import zarr

from scalabel.legacy import label_array_legacy

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PACKAGE_ROOT / "_fixtures" / "inputs"
OUTPUTS_DIR = PACKAGE_ROOT / "_fixtures" / "outputs"


def _make_synthetic() -> np.ndarray:
    """8x8x12 array, tile_shape=(4,4,4) -> 2x2x3 tile grid. Five objects:

    (a) fully-interior single voxel in tile (0,0,0)
    (b) 2-voxel object spanning tiles (0,1,0)-(0,1,1) via a direct face
    (c) 12-voxel chain spanning tiles (1,1,0)-(1,1,1)-(1,1,2) (transitive closure)
    (d1)/(d2) two 2-voxel objects spanning tiles (1,0,0)-(1,0,1) via the SAME
        face at different positions - must NOT be merged with each other
    """
    shape = (8, 8, 12)
    data = np.zeros(shape, dtype=bool)
    data[1, 1, 1] = True
    data[1, 5, 3] = True
    data[1, 5, 4] = True
    data[5, 5, :] = True
    data[5, 1, 3] = True
    data[5, 1, 4] = True
    data[5, 3, 3] = True
    data[5, 3, 4] = True
    return data


def _check_bijection(ref: np.ndarray, mine: np.ndarray, n_ref: int, diagnostics: dict) -> None:
    assert np.array_equal(ref > 0, mine > 0)
    pairs = np.stack([ref.ravel(), mine.ravel()], axis=-1)
    pairs = pairs[pairs[:, 0] != 0]
    unique_pairs = np.unique(pairs, axis=0)
    assert len(unique_pairs) == n_ref, (len(unique_pairs), n_ref)
    assert len(np.unique(unique_pairs[:, 0])) == n_ref
    assert len(np.unique(unique_pairs[:, 1])) == n_ref
    assert diagnostics["n_final_objects"] == n_ref


def _make_zarr(path: Path, data: np.ndarray, dtype, chunks=(2, 2, 2)) -> zarr.Array:
    arr = zarr.create_array(store=str(path), shape=data.shape, chunks=chunks, dtype=dtype, overwrite=True)
    arr[:] = data
    return arr


def test_synthetic_sequential_and_parallel() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data = _make_synthetic()
    tile_shape = (4, 4, 4)  # 2x2x2 storage chunks per tile
    struct = ndi.generate_binary_structure(3, 2)

    ref, n_ref = ndi.label(data, structure=struct)
    assert n_ref == 5

    mask_zarr = _make_zarr(INPUTS_DIR / "synthetic_mask.zarr", data, dtype=np.uint8)

    out_seq = _make_zarr(OUTPUTS_DIR / "synthetic_labels_seq.zarr", np.zeros(data.shape, dtype=np.int64), dtype=np.int64)
    diag_seq = label_array_legacy(mask_zarr, out_seq, tile_shape=tile_shape, connectivity=2, n_workers=1)
    seq_result = out_seq[:]
    _check_bijection(ref, seq_result, n_ref, diag_seq)

    out_par = _make_zarr(OUTPUTS_DIR / "synthetic_labels_par.zarr", np.zeros(data.shape, dtype=np.int64), dtype=np.int64)
    diag_par = label_array_legacy(mask_zarr, out_par, tile_shape=tile_shape, connectivity=2, n_workers=4)
    par_result = out_par[:]
    _check_bijection(ref, par_result, n_ref, diag_par)

    assert np.array_equal(seq_result, par_result)

    # object (c) is a 3-tile chain -> at least one multi-piece boundary group
    assert diag_seq["n_boundary_pieces"] > 0
    assert diag_seq["n_edges"] > 0
    assert diag_seq["n_boundary_groups"] >= 1


if __name__ == "__main__":
    test_synthetic_sequential_and_parallel()
    print("OK")
