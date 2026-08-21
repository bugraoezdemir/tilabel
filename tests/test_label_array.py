"""Correctness tests for the primary API tilewise_ccl.label_array.

`label_array` returns a lazy `dask.array.Array`. For each case we compute it
and assert an exact label-to-label bijection against a direct
`scipy.ndimage.label` reference on the same (tile-free) array - the same
strict ground-truth check, plus a check that the
diagnostic object count matches. Both the sequential (n_workers=1) and
threaded (n_workers>1) Phase-A paths are exercised, and 2D + 3D to confirm the
N-D generality.
"""

import numpy as np
import scipy.ndimage as ndi

from tilewise_ccl import label_array


def _check_bijection(ref: np.ndarray, mine: np.ndarray, n_ref: int, diag: dict) -> None:
    assert np.array_equal(ref > 0, mine > 0)
    pairs = np.stack([ref.ravel(), mine.ravel()], axis=-1)
    pairs = pairs[pairs[:, 0] != 0]
    unique_pairs = np.unique(pairs, axis=0)
    assert len(unique_pairs) == n_ref, (len(unique_pairs), n_ref)
    assert len(np.unique(unique_pairs[:, 0])) == n_ref
    assert len(np.unique(unique_pairs[:, 1])) == n_ref
    assert diag["n_final_objects"] == n_ref
    # labels must be DENSE 1..N (max == N), not just N distinct values - a
    # non-dense output (gaps from halo-only components) would still pass the
    # distinct-count checks above but break offset-based downstream use.
    assert int(mine.max()) == n_ref, ("labels not dense 1..N", int(mine.max()), n_ref)


def _make_3d() -> np.ndarray:
    """8x8x12, tile_shape=(4,4,4) -> 2x2x3 grid. Five objects incl. a 3-tile
    chain (transitive closure) and two objects sharing one face (must stay
    separate)."""
    data = np.zeros((8, 8, 12), dtype=np.uint8)
    data[1, 1, 1] = 1
    data[1, 5, 3] = 1
    data[1, 5, 4] = 1
    data[5, 5, :] = 1
    data[5, 1, 3] = 1
    data[5, 1, 4] = 1
    data[5, 3, 3] = 1
    data[5, 3, 4] = 1
    return data


def _make_2d() -> np.ndarray:
    """16x24, tile_shape=(8,8) -> 2x3 grid. One interior object plus two that
    each straddle a tile boundary."""
    data = np.zeros((16, 24), dtype=np.uint8)
    data[1, 1] = 1
    data[2:5, 7:9] = 1
    data[9:14, 10:14] = 1
    return data


def test_label_array_3d_seq_and_parallel() -> None:
    data = _make_3d()
    struct = ndi.generate_binary_structure(3, 2)
    ref, n_ref = ndi.label(data, structure=struct)
    assert n_ref == 5

    out_seq, diag_seq = label_array(data, tile_shape=(4, 4, 4), connectivity=2, n_workers=1, diagnostics=True)
    _check_bijection(ref, out_seq.compute(), n_ref, diag_seq)

    out_par, diag_par = label_array(data, tile_shape=(4, 4, 4), connectivity=2, n_workers=4, diagnostics=True)
    par_result = out_par.compute()
    _check_bijection(ref, par_result, n_ref, diag_par)

    # a 3-tile chain means at least one multi-piece boundary group
    assert diag_seq["n_boundary_pieces"] > 0
    assert diag_seq["n_edges"] > 0
    assert diag_seq["n_boundary_groups"] >= 1


def test_label_array_2d() -> None:
    data = _make_2d()
    struct = ndi.generate_binary_structure(2, 2)
    ref, n_ref = ndi.label(data, structure=struct)
    assert n_ref == 3

    out, diag = label_array(data, tile_shape=(8, 8), connectivity=2, n_workers=2, diagnostics=True)
    _check_bijection(ref, out.compute(), n_ref, diag)


def test_default_returns_array_only() -> None:
    """By default (diagnostics=False) only the labeled dask array is returned;
    passing diagnostics=True returns a (labels, diag) tuple with identical labels."""
    import dask.array as da

    data = _make_2d()
    out = label_array(data, tile_shape=(8, 8), connectivity=2)
    assert isinstance(out, da.Array), "default should return a dask array, not a tuple"

    out2, diag = label_array(data, tile_shape=(8, 8), connectivity=2, diagnostics=True)
    assert isinstance(diag, dict) and "n_final_objects" in diag
    assert np.array_equal(out.compute(), out2.compute())


def test_properties_area_bbox() -> None:
    """properties=True yields per-object area + bbox matching a whole-array
    scipy computation, and aggregates correctly across tile boundaries."""
    data = _make_3d()  # has objects spanning 2-3 tiles at tile_shape (4,4,4)
    out, diag = label_array(data, tile_shape=(4, 4, 4), connectivity=2, properties=True)
    arr = out.compute()
    n = diag["n_final_objects"]
    assert np.array_equal(diag["label_values"], np.arange(1, n + 1))

    gt_area = np.bincount(arr.ravel(), minlength=n + 1)[1 : n + 1]
    objs = ndi.find_objects(arr, max_label=n)
    gt_lo = np.array([[s[a].start for a in range(3)] for s in objs])
    gt_hi = np.array([[s[a].stop for a in range(3)] for s in objs])
    assert np.array_equal(diag["area"], gt_area)
    assert np.array_equal(diag["bbox_start"], gt_lo)
    assert np.array_equal(diag["bbox_stop"], gt_hi)
    assert int(diag["area"].sum()) == int(data.sum())  # total fg voxels


def test_tile_shape_dim_mismatch_raises() -> None:
    data = _make_2d()
    try:
        label_array(data, tile_shape=(8, 8, 8))
    except ValueError as e:
        assert "dims but mask has" in str(e)
    else:
        raise AssertionError("expected ValueError for tile_shape/mask dim mismatch")


if __name__ == "__main__":
    test_label_array_3d_seq_and_parallel()
    test_label_array_2d()
    test_default_returns_array_only()
    test_properties_area_bbox()
    test_tile_shape_dim_mismatch_raises()
    print("OK")
