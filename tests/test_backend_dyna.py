"""Correctness tests for ``backend="dyna"`` (the dask-free path).

The dyna backend pulls each tile through a ``dyna_zarr`` ``DynamicArray`` and expresses
Phase B as a native dyna ``map_overlap`` instead of a dask graph. It must produce exactly
the same labels as the dask backend and as a whole-array ``scipy.ndimage.label`` - the
backend is an execution choice, never a correctness one.

Skipped when the ``[dyna]`` extra is not installed, so the core suite stays runnable on a
plain ``pip install tilewise-ccl[dev]``.
"""

import numpy as np
import pytest
import scipy.ndimage as ndi

from tilewise_ccl import label_array

dyna_io = pytest.importorskip("dyna_zarr.io", reason="needs the [dyna] extra")
zarr = pytest.importorskip("zarr")

io = dyna_io.io


def _materialize(arr) -> np.ndarray:
    """Both backends' outputs -> numpy (DynamicArray and dask.array both work)."""
    return np.asarray(arr.compute() if hasattr(arr, "compute") else arr)


def _partition(a: np.ndarray):
    """The set of foreground component footprints - identity-independent."""
    return {frozenset(map(tuple, np.argwhere(a == v))) for v in np.unique(a) if v}


def _write(tmp_path, data, chunks, name="src.zarr"):
    p = tmp_path / name
    z = zarr.create_array(store=str(p), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[:] = data
    return str(p)


@pytest.fixture
def blobs():
    """A 3-D volume with compact, well-separated objects plus some tile-crossers."""
    rng = np.random.default_rng(0)
    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    for _ in range(40):
        r = int(rng.integers(3, 9))
        c = rng.integers(r, 64 - r, size=3)
        zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
        sl = tuple(slice(int(ci) - r, int(ci) + r + 1) for ci in c)
        vol[sl][(zz * zz + yy * yy + xx * xx) <= r * r] = 255
    return vol


@pytest.mark.parametrize("connectivity", [1, 2, 3])
def test_matches_scipy(tmp_path, blobs, connectivity):
    """The partition must match a whole-array scipy labeling, for every connectivity."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    expected, _ = ndi.label(blobs > 128,
                            structure=ndi.generate_binary_structure(3, connectivity))
    labels = label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                         connectivity=connectivity, n_workers=2)
    assert _partition(_materialize(labels)) == _partition(expected)


def test_backend_is_inferred(tmp_path, blobs):
    """A DynamicArray mask selects the dyna path without passing `backend`."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    labels = label_array(io.read(path) > 128, tile_shape=(16, 16, 16), connectivity=2)
    assert type(labels).__name__ == "DynamicArray"


def test_dyna_equals_dask(tmp_path, blobs):
    """Same input, both backends -> byte-identical output, not merely equivalent."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    dyna = _materialize(label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                                    connectivity=2, n_workers=2, backend="dyna"))
    dask = _materialize(label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                                    connectivity=2, n_workers=2, backend="dask"))
    np.testing.assert_array_equal(dyna, dask)


@pytest.mark.parametrize("tile_shape", [(16, 16, 16), (32, 32, 32), (24, 24, 24)])
def test_tile_shape_invariance(tmp_path, blobs, tile_shape):
    """Tiling is an execution knob: the partition must not depend on it."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    expected, _ = ndi.label(blobs > 128, structure=ndi.generate_binary_structure(3, 2))
    labels = label_array(io.read(path) > 128, tile_shape=tile_shape,
                         connectivity=2, n_workers=2)
    assert _partition(_materialize(labels)) == _partition(expected)


def test_labels_are_dense(tmp_path, blobs):
    path = _write(tmp_path, blobs, (16, 16, 16))
    out = _materialize(label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                                   connectivity=2, n_workers=2))
    ids = np.unique(out)
    ids = ids[ids > 0]
    assert ids.min() == 1 and ids.max() == ids.size


def test_default_tile_shape(tmp_path, blobs):
    """`tile_shape=None` derives one from the mask, including its chunking."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    expected, _ = ndi.label(blobs > 128, structure=ndi.generate_binary_structure(3, 2))
    labels = label_array(io.read(path) > 128, connectivity=2, n_workers=2)
    assert _partition(_materialize(labels)) == _partition(expected)


def test_properties_match_scipy(tmp_path, blobs):
    """`properties=True` areas/bboxes are metadata - they must still be right."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    labels, diag = label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                               connectivity=2, n_workers=2, properties=True)
    out = _materialize(labels)
    for lbl, area in zip(diag["label_values"], diag["area"]):
        assert int((out == lbl).sum()) == int(area)

    i = int(np.argmax(diag["area"]))
    lbl = int(diag["label_values"][i])
    lo, hi = diag["bbox_start"][i], diag["bbox_stop"][i]
    where = np.argwhere(out == lbl)
    np.testing.assert_array_equal(where.min(axis=0), lo)
    np.testing.assert_array_equal(where.max(axis=0) + 1, hi)


def test_bbox_crop_reads_only_the_box(tmp_path, blobs):
    """The README workflow: locate an object from metadata, then read just its bbox."""
    path = _write(tmp_path, blobs, (16, 16, 16))
    labels, diag = label_array(io.read(path) > 128, tile_shape=(16, 16, 16),
                               connectivity=2, n_workers=2, properties=True)
    i = int(np.argmax(diag["area"]))
    lbl = int(diag["label_values"][i])
    lo, hi = diag["bbox_start"][i], diag["bbox_stop"][i]

    crop = labels[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))].compute()
    assert crop.shape == tuple(int(b - a) for a, b in zip(lo, hi))
    assert int((crop == lbl).sum()) == int(diag["area"][i])


def test_write_round_trip(tmp_path, blobs):
    """Writing the lazy result through dyna's writer must preserve the labels.

    No `region_shape` is passed: the writer picks the tile grid up from the op chain's
    read alignment, which is the behaviour the README documents.
    """
    path = _write(tmp_path, blobs, (16, 16, 16))
    labels = label_array(io.read(path) > 128, tile_shape=(32, 32, 32),
                         connectivity=2, n_workers=2)
    out = tmp_path / "labels.zarr"
    io.write(labels, str(out), chunks=(16, 16, 16), max_workers=2, overwrite=True)

    expected, _ = ndi.label(blobs > 128, structure=ndi.generate_binary_structure(3, 2))
    assert _partition(zarr.open_array(str(out), mode="r")[...]) == _partition(expected)


def test_non_dyna_mask_with_dyna_backend_raises():
    """Explicit backend='dyna' on a numpy mask fails fast, before Phase A."""
    with pytest.raises(TypeError, match="DynamicArray"):
        label_array(np.zeros((8, 8, 8), dtype=bool), tile_shape=(4, 4, 4), backend="dyna")


def test_2d_volume(tmp_path):
    """N-D generality: the dyna path is not 3-D only."""
    rng = np.random.default_rng(3)
    img = (ndi.gaussian_filter(rng.random((96, 96)), 2.0) * 255).astype(np.uint8)
    path = _write(tmp_path, img, (16, 16))
    expected, _ = ndi.label(img > 128, structure=ndi.generate_binary_structure(2, 2))
    labels = label_array(io.read(path) > 128, tile_shape=(16, 16),
                         connectivity=2, n_workers=2)
    assert _partition(_materialize(labels)) == _partition(expected)
