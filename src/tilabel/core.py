"""Tile-wise connected-components labeling with a global boundary-piece graph.

Produces a lazy `dask.array.Array` of final connected-component labels for an
N-D binary mask, via a two-phase approach:

  Phase A (eager): a metadata-only pass over all tiles. Each tile reads a
  1-voxel-haloed block and runs `ndi.label` once, classifying each connected
  piece as INTERIOR (touches none of the tile's 2*ndim faces -> provably a
  complete object) or BOUNDARY (touches a face -> deferred). A per-tile-pair
  face comparison builds a small graph over boundary pieces; a global
  union-find (`reconcile`) groups that graph into final objects. Final ids are
  then assigned DENSELY (1..N: boundary groups first, then interior pieces via
  a running counter) to build a FULL per-tile lookup table (local id -> final
  id, for every local id 1..n). Dense ids let the output fit int32.

  Phase B (lazy): `da.map_overlap` over the input mask. Each block re-labels
  its haloed region (deterministic - reproduces Phase A's local ids exactly)
  and applies that tile's full LUT via fancy indexing, producing the final-id
  block directly - no read-modify-write of any output buffer. The result is a
  genuine `dask.array.Array` that composes with downstream dask operations
  (da.unique, slicing, .to_zarr(), arithmetic, ...).

This is the primary API. For the earlier eager implementation (pre-allocated
output buffer, write-then-read-modify-write Pass 1/Pass 2), see
`tilabel.legacy`.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import scipy.ndimage as ndi

Tile = Tuple[int, ...]
Piece = Tuple[Tile, int]


class UnionFind:
    """Union-find over arbitrary hashable elements, with path compression."""

    def __init__(self) -> None:
        self.parent: Dict[Piece, Piece] = {}
        self.rank: Dict[Piece, int] = {}

    def find(self, x: Piece) -> Piece:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Piece, b: Piece) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def compute_grid_shape(shape: Sequence[int], tile_shape: Sequence[int]) -> Tile:
    return tuple(int(np.ceil(s / t)) for s, t in zip(shape, tile_shape))  # type: ignore[return-value]


def tile_core_slice(tile_idx: Tile, shape: Sequence[int], tile_shape: Sequence[int]) -> Tuple[slice, ...]:
    starts = tuple(idx * t for idx, t in zip(tile_idx, tile_shape))
    stops = tuple(min(s + t, axis_size) for s, t, axis_size in zip(starts, tile_shape, shape))
    return tuple(slice(s, e) for s, e in zip(starts, stops))


def iter_tiles(shape: Sequence[int], tile_shape: Sequence[int]):
    """Yield (tile_idx, core_slice) for every tile, in C order."""
    grid_shape = compute_grid_shape(shape, tile_shape)
    for tile_idx in itertools.product(*[range(n) for n in grid_shape]):
        yield tile_idx, tile_core_slice(tile_idx, shape, tile_shape)


def expand_with_halo(
    core_slice: Tuple[slice, ...], shape: Sequence[int], halo: int = 1
) -> Tuple[Tuple[slice, ...], Tile, Tile]:
    """Expand `core_slice` by `halo` voxels per side, clipped at array bounds.

    Returns (haloed_slice, halo_lo, halo_hi) where halo_lo[axis]/halo_hi[axis] are
    0 or `halo`, indicating whether the halo was actually added on that side
    (it is omitted at the array's outer edges).
    """
    haloed_slices = []
    halo_lo = []
    halo_hi = []
    for s, axis_size in zip(core_slice, shape):
        lo = halo if s.start - halo >= 0 else 0
        hi = halo if s.stop + halo <= axis_size else 0
        haloed_slices.append(slice(s.start - lo, s.stop + hi))
        halo_lo.append(lo)
        halo_hi.append(hi)
    return tuple(haloed_slices), tuple(halo_lo), tuple(halo_hi)


def extract_face(core: np.ndarray, axis: int, side: str) -> np.ndarray:
    idx: List = [slice(None)] * core.ndim
    idx[axis] = 0 if side == "lo" else core.shape[axis] - 1
    return core[tuple(idx)]


def _read_block(arr, slices) -> np.ndarray:
    block = arr[slices]
    if hasattr(block, "compute"):
        block = block.compute()
    return np.asarray(block)


def neighbor_offsets(ndim: int, connectivity: int) -> List[Tuple[int, ...]]:
    """All tile-neighbor direction vectors under `connectivity`.

    A neighbor direction Delta in {-1,0,1}^ndim shares an (ndim -
    count_nonzero(Delta))-dimensional boundary with the tile (a face when 1
    component is nonzero, an edge when 2, a corner when ndim). Two voxels can
    connect across that boundary only if `count_nonzero(Delta) <= connectivity`
    (their minimal crossing offset has exactly those components nonzero), so
    only those directions matter.
    """
    return [
        off
        for off in itertools.product((-1, 0, 1), repeat=ndim)
        if 1 <= sum(1 for v in off if v != 0) <= connectivity
    ]


def _is_canonical(offset: Tuple[int, ...]) -> bool:
    """True for one of each {Delta, -Delta} pair (first nonzero component > 0)."""
    for v in offset:
        if v != 0:
            return v > 0
    return False


def _boundary_slab(core: np.ndarray, offset: Tuple[int, ...]) -> np.ndarray:
    """Core boundary facing `offset`: index nonzero-Delta axes at first/last,
    keep zero-Delta axes full. Copied so it doesn't retain the whole core."""
    idx = tuple(
        (0 if o < 0 else core.shape[d] - 1) if o != 0 else slice(None)
        for d, o in enumerate(offset)
    )
    return np.array(core[idx])


def _tile_metadata(
    tile_idx: Tile,
    core_slice: Tuple[slice, ...],
    mask,
    struct: np.ndarray,
    shape: Sequence[int],
    grid_shape: Tile,
    connectivity: int,
    with_properties: bool = False,
) -> dict:
    """Eager metadata pass for one tile: label, classify interior/boundary.

    Returns small per-tile metadata only (touching ids + cached boundary slabs
    for every neighbor direction under `connectivity`) - never full voxel data -
    so this is safe to call from a thread pool.

    When `with_properties`, also returns this tile's per-local-id voxel count
    (`areas`) and global bounding box (`bbox_lo`/`bbox_hi`, half-open), REUSING
    the labeled `core` already produced above (no second `label` call). These
    come from the tile's core only (tile-bounded) - NOT from whole objects - so
    an object spanning many tiles is never materialized: the caller sums the
    partial areas and unions the partial bboxes across tiles. Memory stays
    bounded by tile size, exactly like the rest of the metadata pass.
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

    all_ids = {int(v) for v in np.unique(core) if v != 0} if n > 0 else set()
    result = {
        "tile_idx": tile_idx,
        "touching_ids": touching_ids,
        "boundaries": boundaries,
        "n_local": n,
        "all_ids": all_ids,
    }

    if with_properties:
        # Two C-optimized reductions over the SAME already-labeled core (the
        # `local`/`core` from the label() call above is reused): bincount for
        # per-id voxel count, find_objects for per-id bbox. Both tile-bounded.
        areas = np.bincount(core.ravel(), minlength=n + 1)[: n + 1].astype(np.int64)
        bbox_lo = np.zeros((n + 1, ndim), dtype=np.int64)
        bbox_hi = np.zeros((n + 1, ndim), dtype=np.int64)
        if n > 0:
            origin = [core_slice[a].start for a in range(ndim)]
            for lid, slc in enumerate(ndi.find_objects(core, max_label=n), start=1):
                if slc is None:
                    continue
                for a in range(ndim):
                    bbox_lo[lid, a] = origin[a] + slc[a].start
                    bbox_hi[lid, a] = origin[a] + slc[a].stop
        result["areas"] = areas
        result["bbox_lo"] = bbox_lo
        result["bbox_hi"] = bbox_hi

    return result


def _transverse_offsets(n_transverse: int, connectivity: int) -> List[Tuple[int, ...]]:
    """Transverse (in-face) offsets for cross-tile adjacency under `connectivity`.

    Two voxels straddling a tile boundary differ by +/-1 along the boundary
    axis (1 nonzero component there) plus some offset `o` in the `n_transverse`
    in-face dimensions. `generate_binary_structure(ndim, connectivity)` includes
    an offset iff its number of nonzero components is <= connectivity, so the
    in-face part must satisfy `count_nonzero(o) <= connectivity - 1`. For
    connectivity=1 this is only the zero offset (aligned faces); for higher
    connectivity it also includes the diagonal/corner crossings.
    """
    offsets: List[Tuple[int, ...]] = []
    for o in itertools.product((-1, 0, 1), repeat=n_transverse):
        if sum(1 for v in o if v != 0) <= connectivity - 1:
            offsets.append(o)
    return offsets


def _aligned_faces(face_a: np.ndarray, face_b: np.ndarray, offset: Tuple[int, ...]):
    """Slice `face_a`/`face_b` so element p of A aligns with element p+offset of B."""
    a_sl, b_sl = [], []
    for od in offset:
        if od == 0:
            a_sl.append(slice(None))
            b_sl.append(slice(None))
        elif od == 1:
            a_sl.append(slice(0, -1))
            b_sl.append(slice(1, None))
        else:  # od == -1
            a_sl.append(slice(1, None))
            b_sl.append(slice(0, -1))
    return face_a[tuple(a_sl)], face_b[tuple(b_sl)]


def build_edges(
    tile_results_by_idx: Dict[Tile, dict], grid_shape: Tile, connectivity: int
) -> List[Tuple[Piece, Piece]]:
    """Link boundary pieces that touch across a shared tile boundary.

    Considers EVERY neighbor direction admissible under `connectivity` (faces,
    and for connectivity > 1 also edge/corner-diagonal tile neighbors), each
    unordered tile pair once. For a direction Delta, tile A's boundary slab
    facing Delta is compared against neighbor B's slab facing -Delta, at the
    transverse (within-boundary) offsets the connectivity still permits
    (count_nonzero(transverse) <= connectivity - count_nonzero(Delta)). Missing
    the diagonal directions would split objects that cross a tile boundary only
    through a corner/edge.
    """
    ndim = len(grid_shape)
    directions = [off for off in neighbor_offsets(ndim, connectivity) if _is_canonical(off)]
    edges: List[Tuple[Piece, Piece]] = []
    for tile_idx, result in tile_results_by_idx.items():
        for off in directions:
            neighbor = tuple(t + o for t, o in zip(tile_idx, off))
            if neighbor not in tile_results_by_idx:
                continue
            slab_a = result["boundaries"].get(off)
            slab_b = tile_results_by_idx[neighbor]["boundaries"].get(tuple(-o for o in off))
            if slab_a is None or slab_b is None:
                continue
            n_delta = sum(1 for o in off if o != 0)
            n_transverse = ndim - n_delta
            # transverse budget: count_nonzero(t) <= connectivity - n_delta
            for offset in _transverse_offsets(n_transverse, connectivity - n_delta + 1):
                fa, fb = _aligned_faces(slab_a, slab_b, offset)
                fa = np.atleast_1d(fa)
                fb = np.atleast_1d(fb)
                if fa.size == 0:
                    continue
                both = (fa != 0) & (fb != 0)
                if not np.any(both):
                    continue
                pairs = np.unique(np.stack([fa[both], fb[both]], axis=-1), axis=0)
                for a_lid, b_lid in pairs:
                    edges.append(((tile_idx, int(a_lid)), (neighbor, int(b_lid))))  # type: ignore[arg-type]
    return edges


def reconcile(
    tile_results: List[dict], edges: List[Tuple[Piece, Piece]], n_tiles: int, big_offset: int
) -> Tuple[Dict[Piece, int], int]:
    """Union-find over boundary pieces -> {(tile_idx, local_id): final_id}."""
    uf = UnionFind()
    for a, b in edges:
        uf.union(a, b)

    all_pieces: set = set()
    for result in tile_results:
        for lid in result["touching_ids"]:
            all_pieces.add((result["tile_idx"], lid))

    groups: Dict[Piece, List[Piece]] = defaultdict(list)
    for piece in all_pieces:
        groups[uf.find(piece)].append(piece)

    lut: Dict[Piece, int] = {}
    next_id = n_tiles * big_offset + 1  # disjoint from every tile's interior-id range
    for pieces in groups.values():
        for piece in pieces:
            lut[piece] = next_id
        next_id += 1

    return lut, len(groups)


def label_array(
    mask,
    tile_shape: Optional[Sequence[int]] = None,
    connectivity: int = 2,
    n_workers: int = 1,
    diagnostics: bool = False,
    properties: bool = False,
    verbose: bool = False,
) -> Union[da.Array, Tuple[da.Array, dict]]:
    """Label connected components of a binary N-D mask into a lazy dask array.

    Parameters
    ----------
    mask : zarr.Array | dask.array.Array | np.ndarray
        N-D array, truthy where foreground.
    tile_shape : sequence of int, optional
        Tile size along each axis; must have the same length as `mask.ndim`.
        `None` (the default) uses `(303,) * mask.ndim`, adapting to the mask's
        dimensionality. Execution hyperparameter, independent of storage chunk
        size but ideally an integer multiple of it.
    connectivity : int
        Passed to `scipy.ndimage.generate_binary_structure(ndim, connectivity)`.
    n_workers : int
        If > 1, Phase A (metadata pass) runs via a thread pool. Phase B is a
        lazy dask graph - its concurrency is controlled by dask's scheduler
        when the caller computes/writes the result.
    diagnostics : bool
        If True, also return a `diag` dict of tile/object/graph counts and
        timings (see below). Default False - only the labeled array is returned.
    properties : bool
        If True, additionally compute per-object `label_values`, `area` (voxel
        count) and `bbox_start`/`bbox_stop` (half-open global bounding box per
        axis) and put them in `diag` (implies returning the `diag` tuple). These
        are aggregated as metadata across tiles - a small extra per-tile
        reduction (bincount + find_objects on the already-labeled core), memory
        still bounded by tile size regardless of object size.
    verbose : bool
        If True, print per-tile progress for Phase A (eagerly) and Phase B
        (when the returned array is computed/written).

    Returns
    -------
    output_labels : dask.array.Array
        A lazy array of DENSE final labels 1..N (background 0), same shape as
        `mask`. Its dtype is int32 when N < 2**31 (the usual case) else int64.
        Returned directly when `diagnostics` is False (the default).
    (output_labels, diag) : (dask.array.Array, dict)
        When `diagnostics` (or `properties`) is True, a tuple is returned
        instead: `diag` holds tile/object/graph counts (object tile-span
        distribution, boundary-graph sizes, `output_dtype`, ...) and eager-phase
        timings. With `properties`, `diag` also carries `label_values` (1..N),
        `area` (voxels per object), and `bbox_start`/`bbox_stop` ((N, ndim)
        half-open global bounding box per object). Phase B's actual cost is
        incurred later, on compute/write.
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

    # --- Phase A (eager): metadata-only pass ---
    def _meta(item):
        tile_idx, core_slice = item
        result = _tile_metadata(tile_idx, core_slice, mask, struct, shape, grid_shape, connectivity, properties)
        if verbose:
            with print_lock:
                progress["done"] += 1
                n_interior = len(result["all_ids"] - result["touching_ids"])
                print(
                    f"  [Phase A {progress['done']}/{n_tiles}] tile {tile_idx}: "
                    f"n_local={result['n_local']}, interior={n_interior}, "
                    f"boundary={len(result['touching_ids'])}"
                )
        return result

    t0 = time.perf_counter()
    if verbose:
        print(f"Phase A: computing metadata for {n_tiles} tiles (grid_shape={grid_shape})...")
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            tile_results = list(ex.map(_meta, tiles))
    else:
        tile_results = [_meta(item) for item in tiles]
    t1 = time.perf_counter()

    tile_results_by_idx = {r["tile_idx"]: r for r in tile_results}
    edges = build_edges(tile_results_by_idx, grid_shape, connectivity)
    lut, n_groups = reconcile(tile_results, edges, n_tiles, big_offset)
    t2 = time.perf_counter()
    if verbose:
        print(f"Reconciliation: {len(edges)} edges -> {n_groups} boundary groups")

    # --- Assign DENSE final ids (1..N) and build a FULL per-tile LUT.
    # `reconcile` returns SPARSE boundary-group ids (a disjoint high range,
    # shared with the legacy path's collision-free scheme); we remap them here
    # to a dense 1..n_groups range and give interior pieces the ids
    # n_groups+1..N. This is a metadata-only loop over PIECES (not voxels), so
    # producing dense ids costs nothing extra - no compaction pass over the
    # array - and lets the output fit int32 whenever N < 2**31, halving the
    # write volume vs sparse int64 ids. ---
    n_interior_total = sum(len(r["all_ids"] - r["touching_ids"]) for r in tile_results)
    n_final = n_interior_total + n_groups
    out_dtype = np.int32 if n_final <= np.iinfo(np.int32).max else np.int64

    group_dense: Dict[int, int] = {}  # sparse reconcile id -> dense id (1..n_groups)
    next_interior = n_groups + 1
    full_luts: Dict[Tile, np.ndarray] = {}
    for tile_idx, result in tile_results_by_idx.items():
        n = result["n_local"]
        touching = result["touching_ids"]
        all_ids = result["all_ids"]  # lids actually present in this tile's CORE
        full_lut = np.zeros(n + 1, dtype=out_dtype)
        for lid in range(1, n + 1):
            if lid in touching:
                sparse = lut[(tile_idx, lid)]
                dense = group_dense.get(sparse)
                if dense is None:
                    dense = len(group_dense) + 1
                    group_dense[sparse] = dense
                full_lut[lid] = dense
            elif lid in all_ids:
                # interior core piece -> a fresh dense id
                full_lut[lid] = next_interior
                next_interior += 1
            # else: halo-only component (in the haloed block but not in this
            # tile's core) - trimmed away by map_overlap, so map it to 0 and do
            # NOT consume an id (otherwise the output would be non-dense).
        full_luts[tile_idx] = full_lut

    n_pass2_tiles = sum(1 for r in tile_results if r["touching_ids"])

    diag = None
    if diagnostics or properties:
        # --- Tile-span distribution of objects (metadata-only, no data pass).
        # Interior objects live in exactly 1 tile. Each boundary GROUP (union-find
        # over boundary pieces) is one object; the number of DISTINCT tiles its
        # pieces occupy is how many tiles that object spans. A group spanning 1
        # tile touches a border but does not cross it (its neighbor was
        # background); spanning >=2 tiles means it crosses tile border(s);
        # spanning >3 tiles is a "large" object straddling many tiles. Edges only
        # ever link pieces in DIFFERENT tiles, so a 1-tile group has one piece. ---
        group_tiles: Dict[int, set] = defaultdict(set)
        for (piece_tile, _lid), sparse_id in lut.items():
            group_tiles[sparse_id].add(piece_tile)
        tile_spans = [len(ts) for ts in group_tiles.values()]
        n_groups_single = sum(1 for c in tile_spans if c == 1)
        n_crossing = sum(1 for c in tile_spans if c >= 2)
        n_crossing_gt3 = sum(1 for c in tile_spans if c > 3)
        max_tiles_spanned = max(tile_spans) if tile_spans else (1 if n_interior_total else 0)

        diag = {
            "n_tiles": n_tiles,
            "grid_shape": grid_shape,
            "n_interior_objects": n_interior_total,
            "n_boundary_pieces": sum(len(r["touching_ids"]) for r in tile_results),
            "n_edges": len(edges),
            "n_boundary_groups": n_groups,
            "n_final_objects": n_final,
            "output_dtype": np.dtype(out_dtype).name,
            # object tile-span distribution (understanding large/crossing objects)
            "n_objects_single_tile": n_interior_total + n_groups_single,
            "n_objects_crossing": n_crossing,
            "n_objects_crossing_gt3_tiles": n_crossing_gt3,
            "max_tiles_spanned": max_tiles_spanned,
            "time_phaseA_s": t1 - t0,
            "time_reconcile_s": t2 - t1,
        }

        if properties:
            # --- Per-object area + bbox, aggregated from per-tile partials
            # (metadata only). Each tile contributed voxel counts and a local
            # bbox per local id; map those to final ids via full_lut and combine:
            # SUM areas, UNION bboxes. Vectorized per tile (loop over tiles, not
            # objects). Never touches whole objects -> memory ~ O(n_objects). ---
            ndim = len(shape)
            area = np.zeros(n_final + 1, dtype=np.int64)
            bbox_lo = np.full((n_final + 1, ndim), np.iinfo(np.int64).max, dtype=np.int64)
            bbox_hi = np.zeros((n_final + 1, ndim), dtype=np.int64)
            for tile_idx, result in tile_results_by_idx.items():
                n = result["n_local"]
                if n == 0:
                    continue
                finals = full_luts[tile_idx][1 : n + 1]  # final id per local id (0 = halo-only)
                valid = finals > 0
                if not valid.any():
                    continue
                fv = finals[valid]
                np.add.at(area, fv, result["areas"][1 : n + 1][valid])
                np.minimum.at(bbox_lo, fv, result["bbox_lo"][1 : n + 1][valid])
                np.maximum.at(bbox_hi, fv, result["bbox_hi"][1 : n + 1][valid])
            diag["label_values"] = np.arange(1, n_final + 1, dtype=np.int64)
            diag["area"] = area[1:]
            diag["bbox_start"] = bbox_lo[1:]
            diag["bbox_stop"] = bbox_hi[1:]

    # --- Phase B (lazy): map_overlap re-labels + applies the per-tile LUT ---
    if hasattr(mask, "rechunk"):
        mask_da = mask.rechunk(tile_shape)
    else:
        mask_da = da.from_array(mask, chunks=tile_shape)

    progress2 = {"done": 0}

    def _apply_block(block: np.ndarray, block_info=None) -> np.ndarray:
        tile_idx = tuple(block_info[0]["chunk-location"])
        local, _ = ndi.label(block > 0, structure=struct)
        out = full_luts[tile_idx][local]
        if verbose:
            n_boundary = len(tile_results_by_idx[tile_idx]["touching_ids"])
            if n_boundary:
                with print_lock:
                    progress2["done"] += 1
                    print(
                        f"  [Phase B {progress2['done']}/{n_pass2_tiles}] tile {tile_idx}: "
                        f"applied LUT to {n_boundary} boundary piece(s)"
                    )
        return out

    if verbose:
        print(
            f"Phase B: lazy map_overlap built ({n_pass2_tiles}/{n_tiles} tiles with "
            f"boundary pieces) - prints below occur on .compute()/.to_zarr()"
        )

    output_labels = da.map_overlap(
        _apply_block,
        mask_da,
        depth=1,
        boundary="none",
        trim=True,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    if diagnostics or properties:
        diag["time_total_s"] = time.perf_counter() - t0
        return output_labels, diag
    return output_labels
