"""`tile_shape=None` derives a tile from the mask instead of a hard-coded constant.

The old default was `(303,) * ndim`, which ignored the array's shape, its chunking, and
its dimensionality: 106 MiB of int32 labels at 3-D, but 32 GiB at 4-D and ~9.7 TiB at 5-D,
and it tiled the t/c axes of an OME-Zarr array by a spatial constant. The replacement
targets a memory budget over the trailing (spatial) axes and snaps to whole storage chunks.
"""

import numpy as np
import pytest
import scipy.ndimage as ndi

from tilewise_ccl import label_array
from tilewise_ccl.core import default_tile_shape, _DEFAULT_TILE_MIB, _SPATIAL_AXES


def _mib(tile):
    return float(np.prod(tile)) * 4 / 1024**2


@pytest.mark.parametrize("shape,chunks", [
    ((2048, 2048, 2048), (64, 64, 64)),
    ((6000, 6000, 6000), (96, 96, 96)),
    ((4096, 4096), (256, 256)),
    ((10, 3, 512, 512, 512), (1, 1, 64, 64, 64)),
])
def test_within_budget(shape, chunks):
    """The chosen tile must not exceed the target working set."""
    assert _mib(default_tile_shape(shape, chunks)) <= _DEFAULT_TILE_MIB + 1e-6


@pytest.mark.parametrize("shape,chunks", [
    ((2048, 2048, 2048), (64, 64, 64)),
    ((2048, 2048, 2048), (96, 96, 96)),
    ((10, 3, 512, 512, 512), (1, 1, 64, 64, 64)),
])
def test_is_whole_chunk_multiple(shape, chunks):
    """Every axis is a whole chunk multiple (or the full extent), so no chunk is
    read by two tiles."""
    tile = default_tile_shape(shape, chunks)
    for t, c, s in zip(tile, chunks, shape):
        assert t == s or t % c == 0, f"tile {tile} not a chunk multiple of {chunks}"


def test_never_exceeds_array():
    tile = default_tile_shape((100, 100, 100), (64, 64, 64))
    assert tile == (100, 100, 100)


def test_leading_axes_are_not_tiled():
    """A 5-D (t, c, z, y, x) array must be tiled over z/y/x only."""
    tile = default_tile_shape((10, 3, 512, 512, 512), (1, 1, 64, 64, 64))
    assert tile[0] == 1 and tile[1] == 1
    assert all(t > 1 for t in tile[-_SPATIAL_AXES:])


def test_scales_with_dimensionality():
    """The old constant exploded with ndim; the budget must hold across 2/3/4/5-D."""
    for shape, chunks in [((4096, 4096), (64, 64)),
                          ((1024,) * 3, (64,) * 3),
                          ((4, 512, 512, 512), (1, 64, 64, 64)),
                          ((4, 2, 512, 512, 512), (1, 1, 64, 64, 64))]:
        assert _mib(default_tile_shape(shape, chunks)) <= _DEFAULT_TILE_MIB + 1e-6


def test_works_without_chunks():
    """A mask with no `.chunks` still gets a budgeted tile."""
    tile = default_tile_shape((2048, 2048, 2048), None)
    assert _mib(tile) <= _DEFAULT_TILE_MIB + 1e-6
    assert all(t > 1 for t in tile)


def test_default_labels_correctly():
    """`tile_shape=None` must produce the same partition as an explicit tile."""
    vol = ndi.gaussian_filter(np.random.default_rng(9).random((80, 80, 80)), 1.5) > 0.53
    expected, _ = ndi.label(vol, structure=ndi.generate_binary_structure(3, 2))

    def part(a):
        return {frozenset(map(tuple, np.argwhere(a == v))) for v in np.unique(a) if v}

    got = np.asarray(label_array(vol, connectivity=2, n_workers=2))
    assert part(got) == part(expected)


def test_accepts_dask_style_chunks():
    """A dask array reports `.chunks` as a tuple-of-tuples, zarr as flat ints.

    `int(c)` on the tuple raised `TypeError: int() argument must be ... not 'tuple'`
    for every dask-backed input to `label_array`.
    """
    flat = default_tile_shape((1, 1, 256, 256, 256), (1, 1, 64, 64, 64))
    nested = default_tile_shape(
        (1, 1, 256, 256, 256),
        ((1,), (1,), (64, 64, 64, 64), (64, 64, 64, 64), (64, 64, 64, 64)),
    )
    assert nested == flat


def test_ragged_last_chunk_uses_the_regular_size():
    """The trailing block is short; the tile must key off the regular block size."""
    got = default_tile_shape(
        (1, 1, 200, 200, 200),
        ((1,), (1,), (64, 64, 64, 8), (64, 64, 64, 8), (64, 64, 64, 8)),
    )
    assert all(g > 0 for g in got)
    assert got == default_tile_shape((1, 1, 200, 200, 200), (1, 1, 64, 64, 64))
