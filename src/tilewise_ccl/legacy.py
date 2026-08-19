"""Eager tile-wise connected-components labeling into a pre-allocated output.

LEGACY: superseded by `tilewise_ccl.label_array`, which produces an equivalent
(exact-match) lazy `dask.array.Array` directly - no pre-allocated output
buffer and no Pass 2 read-modify-write - and is ~2.5-3.5x faster end-to-end.
Kept for comparison/benchmarking.

Pass 1: a tile-local `ndimage.label` call classifies each connected piece as
INTERIOR (touches none of the tile's 2*ndim faces -> provably a complete
object, written immediately) or BOUNDARY (touches a face -> deferred). A
1-voxel halo + per-tile-pair face comparison builds a small graph over
boundary pieces; a global union-find groups that graph into final objects.

Pass 2: re-labels each tile with a boundary piece (deterministic), reads back
its core region from `output_labels`, applies the LUT to boundary pixels, and
writes the result back.

Pass 1 and Pass 2 are tile-local (each tile reads/writes only its own
non-overlapping core region of `output_labels`) and can run sequentially or
via a thread pool.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi

from tilewise_ccl.core import (
    Piece,
    Tile,
    _boundary_slab,
    _read_block,
    build_edges,
    compute_grid_shape,
    expand_with_halo,
    iter_tiles,
    neighbor_offsets,
    reconcile,
)


def process_tile(
    tile_idx: Tile,
    core_slice: Tuple[slice, ...],
    mask,
    struct: np.ndarray,
    shape: Sequence[int],
    grid_shape: Tile,
    big_offset: int,
    connectivity: int,
    output_labels,
) -> dict:
    """Pass 1 for one tile: label, classify interior/boundary, write interior pieces.

    Returns small per-tile metadata only (touching ids + cached boundary slabs
    for every neighbor direction under `connectivity`) - never full voxel data -
    so this is safe to call from a thread pool.
    """
    haloed_slice, halo_lo, _ = expand_with_halo(core_slice, shape, halo=1)
    haloed = _read_block(mask, haloed_slice) > 0
    local, n = ndi.label(haloed, structure=struct)

    ndim = len(core_slice)
    core_shape = tuple(s.stop - s.start for s in core_slice)
    core = local[tuple(slice(halo_lo[a], halo_lo[a] + core_shape[a]) for a in range(ndim))]

    boundaries: Dict[Tuple[int, ...], np.ndarray] = {}
    touching_ids: set = set()
    if n > 0:
        for off in neighbor_offsets(ndim, connectivity):
            neighbor = tuple(t + o for t, o in zip(tile_idx, off))
            if not all(0 <= neighbor[a] < grid_shape[a] for a in range(ndim)):
                continue
            slab = _boundary_slab(core, off)
            ids = {int(v) for v in np.unique(slab) if v != 0}
            if ids:
                boundaries[off] = slab
                touching_ids |= ids

    n_interior = 0
    if n == 0:
        return {
            "tile_idx": tile_idx,
            "touching_ids": touching_ids,
            "boundaries": boundaries,
            "n_interior": 0,
            "n_local": 0,
        }

    all_ids = {int(v) for v in np.unique(core) if v != 0}
    interior_ids = all_ids - touching_ids
    n_interior = len(interior_ids)

    tile_linear = int(np.ravel_multi_index(tile_idx, grid_shape))
    id_offset = tile_linear * big_offset

    if interior_ids:
        lut_interior = np.zeros(n + 1, dtype=np.int64)
        for lid in interior_ids:
            lut_interior[lid] = id_offset + lid
        out_block = lut_interior[core]
        output_labels[core_slice] = out_block

    return {
        "tile_idx": tile_idx,
        "touching_ids": touching_ids,
        "boundaries": boundaries,
        "n_interior": n_interior,
        "n_local": n,
    }


def apply_tile(
    tile_idx: Tile,
    core_slice: Tuple[slice, ...],
    mask,
    struct: np.ndarray,
    shape: Sequence[int],
    touching_ids: set,
    lut: Dict[Piece, int],
    output_labels,
) -> None:
    """Pass 2 for one tile: re-label (deterministic) and apply the LUT to boundary pieces."""
    haloed_slice, halo_lo, _ = expand_with_halo(core_slice, shape, halo=1)
    haloed = _read_block(mask, haloed_slice) > 0
    local, n = ndi.label(haloed, structure=struct)

    core_shape = tuple(s.stop - s.start for s in core_slice)
    core = local[tuple(slice(halo_lo[a], halo_lo[a] + core_shape[a]) for a in range(len(core_slice)))]

    lut_array = np.zeros(n + 1, dtype=np.int64)
    for lid in touching_ids:
        lut_array[lid] = lut[(tile_idx, lid)]

    out_block = _read_block(output_labels, core_slice)
    mapped = lut_array[core]
    boundary_mask = mapped != 0
    out_block[boundary_mask] = mapped[boundary_mask]
    output_labels[core_slice] = out_block


def label_array_legacy(
    mask,
    output_labels,
    tile_shape: Optional[Sequence[int]] = None,
    connectivity: int = 2,
    n_workers: int = 1,
    verbose: bool = False,
) -> dict:
    """Label connected components of a binary N-D mask into `output_labels`.

    LEGACY: prefer `tilewise_ccl.label_array` (returns a lazy `dask.array.Array`
    instead of writing into a pre-allocated buffer, with the same object
    identities and ~2.5-3.5x faster end-to-end). This implementation is kept
    for comparison/benchmarking.

    Parameters
    ----------
    mask : zarr.Array | dask.array.Array | np.ndarray
        N-D array, truthy where foreground.
    output_labels : zarr.Array | np.ndarray
        Pre-allocated N-D int64 array of the same shape, written in place.
    tile_shape : sequence of int
        Tile size along each axis (must have the same length as `mask.ndim`).
        Execution hyperparameter, independent of storage chunk size but
        ideally an integer multiple of it.
    connectivity : int
        Passed to `scipy.ndimage.generate_binary_structure(ndim, connectivity)`.
    n_workers : int
        If > 1, Pass 1 and Pass 2 run via a thread pool (each tile touches only
        its own non-overlapping core region, so this is safe).
    verbose : bool
        If True, print per-tile progress for Pass 1 and Pass 2.

    Returns
    -------
    dict of diagnostics (tile/object/graph counts).
    """
    shape = tuple(int(s) for s in mask.shape)
    ndim = len(shape)
    if tile_shape is None:
        tile_shape = (303,) * ndim  # default adapts to the mask's dimensionality
    elif len(tile_shape) != ndim:
        raise ValueError(f"tile_shape has {len(tile_shape)} dims but mask has {ndim}")
    struct = ndi.generate_binary_structure(ndim, connectivity)
    grid_shape = compute_grid_shape(shape, tile_shape)
    n_tiles = int(np.prod(grid_shape))
    big_offset = int(np.prod([t + 2 for t in tile_shape]))  # safe upper bound on n per tile

    tiles = list(iter_tiles(shape, tile_shape))

    print_lock = threading.Lock()
    progress = {"done": 0}

    def _pass1(item):
        tile_idx, core_slice = item
        result = process_tile(tile_idx, core_slice, mask, struct, shape, grid_shape, big_offset, connectivity, output_labels)
        if verbose:
            with print_lock:
                progress["done"] += 1
                print(
                    f"  [Pass 1 {progress['done']}/{n_tiles}] tile {tile_idx}: "
                    f"n_local={result['n_local']}, interior={result['n_interior']}, "
                    f"boundary={len(result['touching_ids'])}"
                )
        return result

    t0 = time.perf_counter()
    if verbose:
        print(f"Pass 1: labeling {n_tiles} tiles (grid_shape={grid_shape})...")
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            tile_results = list(ex.map(_pass1, tiles))
    else:
        tile_results = [_pass1(item) for item in tiles]
    t1 = time.perf_counter()

    tile_results_by_idx = {r["tile_idx"]: r for r in tile_results}
    edges = build_edges(tile_results_by_idx, grid_shape, connectivity)
    lut, n_groups = reconcile(tile_results, edges, n_tiles, big_offset)
    t2 = time.perf_counter()
    if verbose:
        print(f"Reconciliation: {len(edges)} edges -> {n_groups} boundary groups")

    n_pass2_tiles = sum(1 for r in tile_results if r["touching_ids"])
    progress["done"] = 0

    def _pass2(item):
        tile_idx, core_slice = item
        result = tile_results_by_idx[tile_idx]
        if result["touching_ids"]:
            apply_tile(tile_idx, core_slice, mask, struct, shape, result["touching_ids"], lut, output_labels)
            if verbose:
                with print_lock:
                    progress["done"] += 1
                    print(
                        f"  [Pass 2 {progress['done']}/{n_pass2_tiles}] tile {tile_idx}: "
                        f"applied LUT to {len(result['touching_ids'])} boundary piece(s)"
                    )

    if verbose:
        print(f"Pass 2: applying LUT to {n_pass2_tiles} tiles with boundary pieces...")
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_pass2, tiles))
    else:
        for item in tiles:
            _pass2(item)
    t3 = time.perf_counter()

    n_interior_total = sum(r["n_interior"] for r in tile_results)
    n_boundary_pieces = sum(len(r["touching_ids"]) for r in tile_results)

    return {
        "n_tiles": n_tiles,
        "grid_shape": grid_shape,
        "n_interior_objects": n_interior_total,
        "n_boundary_pieces": n_boundary_pieces,
        "n_edges": len(edges),
        "n_boundary_groups": n_groups,
        "n_final_objects": n_interior_total + n_groups,
        "time_pass1_s": t1 - t0,
        "time_reconcile_s": t2 - t1,
        "time_pass2_s": t3 - t2,
        "time_total_s": t3 - t0,
    }
