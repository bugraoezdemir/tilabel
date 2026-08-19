"""Tile-wise connected-components labeling with a global boundary-piece graph.

Produces a lazy `dask.array.Array` of final connected-component labels for an
N-D binary mask, via a two-phase approach:

  Phase A (eager): a metadata-only pass over all tiles. Each tile reads its
  OWN core (no halo) and runs `ndi.label` once, classifying each connected
  piece as INTERIOR (touches none of the tile's 2*ndim faces -> provably a
  complete object) or BOUNDARY (touches a face -> deferred). A per-tile-pair
  face comparison builds a small graph over boundary pieces; a global
  union-find (`reconcile`) groups that graph into final objects. Final ids are
  then assigned DENSELY (1..N: boundary groups first, then interior pieces via
  a running counter) to build a FULL per-tile lookup table (local id -> final
  id, for every local id 1..n). Dense ids let the output fit int32.

  Phase B (lazy): `da.map_blocks` over the input mask. Each block re-labels
  its own core (deterministic - reproduces Phase A's local ids exactly) and
  applies that tile's full LUT via fancy indexing, producing the final-id
  block directly - no read-modify-write of any output buffer. The result is a
  genuine `dask.array.Array` that composes with downstream dask operations
  (da.unique, slicing, .to_zarr(), arithmetic, ...).

NO HALO: labeling each tile's core in isolation is sufficient, because the
boundary graph + union-find already merges pieces that are connected only
through a neighbouring tile (two such pieces both touch the shared face, so
both link to the neighbour's piece and land in one group). A 1-voxel halo would
only pre-merge some of those pairs locally - the same final partition, at the
cost of reading/labelling a larger block, `map_overlap`'s cross-block machinery,
and having to discard "halo-only" components. Verified identical to a whole-array
`scipy.ndimage.label` for connectivity 1/2/3.

This is the primary API. For the earlier eager implementation (pre-allocated
output buffer, write-then-read-modify-write Pass 1/Pass 2), see
`tilewise_ccl.legacy`.
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import scipy.ndimage as ndi

Tile = Tuple[int, ...]
Piece = Tuple[Tile, int]

#: scipy `generate_binary_structure` rank -> cc3d's 3-D connectivity number. cc3d is a
#: dedicated C++ CCL library: ~2x faster than `ndi.label` per call but it HOLDS THE GIL, so
#: it only pays off with `executor='process'` (with threads it is ~4x SLOWER overall, since
#: `ndi.label` releases the GIL and scales across the pool). Verified to produce byte-identical
#: label ids to `ndi.label` for ranks 1/2/3, so the two are interchangeable between phases.
_CC3D_CONN_3D = {1: 6, 2: 18, 3: 26}


def _label_block(blk, struct, connectivity, labeler="scipy"):
    """Connected-component label one block -> (labels, n). `labeler='cc3d'` uses the cc3d
    library (3-D only; falls back to scipy otherwise), else `scipy.ndimage.label`."""
    if labeler == "cc3d" and blk.ndim == 3:
        import cc3d
        return cc3d.connected_components(
            blk, connectivity=_CC3D_CONN_3D[connectivity], return_N=True)
    return ndi.label(blk, structure=struct)


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


#: Target working-set size for an auto-chosen tile, in MiB (the labeled int32 core).
#: MEASURED: real peak RSS is ~2x this per tile (mask + labels + scipy internals), and
#: Phase A holds `n_workers` tiles at once - so 256 MiB is ~0.5 GiB per worker, ~4 GiB at
#: n_workers=8. Large enough that most objects fall entirely inside one tile (which is what
#: keeps the boundary graph small), while still safe unattended on a 16 GB machine. Pass
#: `tile_shape` explicitly to trade memory for fewer, larger tiles.
_DEFAULT_TILE_MIB = 256

#: How many trailing axes are treated as SPATIAL when auto-choosing a tile. An OME-Zarr
#: array is typically (t, c, z, y, x): tiling t/c by a spatial constant is meaningless, so
#: leading axes stay at 1 and only the trailing 3 (z, y, x) are grown. Mirrors
#: `ome_zarr_pyramid.utils.array_utils.autocompute_chunk_shape`, which does the same via
#: named axes - dyna has no axis names, so position stands in.
_SPATIAL_AXES = 3


def default_tile_shape(shape: Sequence[int], chunks=None,
                       target_mib: float = _DEFAULT_TILE_MIB) -> Tile:
    """Pick a tile shape for `shape` under a ~`target_mib` int32 working-set budget.

    Seeds the trailing `_SPATIAL_AXES` axes isotropically (the cube root of the voxel
    budget) and grows them until the budget is spent; leading axes stay at 1, so a 5-D
    (t, c, z, y, x) array is tiled over z/y/x only. When `chunks` is given, every axis is
    seeded and grown in whole chunk multiples, so the tile is a multiple of the storage
    chunk by construction - no chunk is ever read by two tiles.

    Replaces a hard-coded ``(303,) * ndim``, which ignored the array shape, its chunking,
    and its dimensionality: 106 MiB of int32 at 3-D but 32 GiB at 4-D and ~9.7 TiB at 5-D.
    """
    ndim = len(shape)
    shape = [int(s) for s in shape]
    budget = max(1, int(target_mib * 1024 * 1024) // 4)      # voxels, at int32 labels

    if chunks is not None and len(chunks) == ndim:
        step = [max(1, int(c)) for c in chunks]
    else:
        step = [1] * ndim

    spatial = list(range(max(0, ndim - _SPATIAL_AXES), ndim))
    tile = [min(step[a], shape[a]) for a in range(ndim)]

    # Seed the spatial axes at an isotropic side, snapped DOWN to a whole step.
    if spatial:
        side = int(budget ** (1.0 / len(spatial)))
        for a in spatial:
            k = max(1, side // step[a])
            tile[a] = min(shape[a], k * step[a])

    # Spend whatever budget is left, one step at a time, cycling the spatial axes so the
    # tile stays as isotropic as the chunking allows. Trailing axes grow first (contiguous).
    growable = True
    while growable:
        growable = False
        for a in reversed(spatial):
            if tile[a] >= shape[a]:
                continue
            nxt = min(shape[a], tile[a] + step[a])
            trial = tile[:a] + [nxt] + tile[a + 1:]
            if int(np.prod(trial)) <= budget:
                tile[a] = nxt
                growable = True

    return tuple(int(t) for t in tile)


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


def _boundary_slab(core: np.ndarray, offset: Tuple[int, ...], n_local: int = 0) -> np.ndarray:
    """Core boundary facing `offset`: index nonzero-Delta axes at first/last,
    keep zero-Delta axes full. Copied so it doesn't retain the whole core.

    Stored in the NARROWEST unsigned dtype that holds this tile's local ids (`n_local`).
    These slabs are the dominant Phase-A memory: the 6 face slabs of a tile are ~t^2 each,
    so at a 672^3 tile they are 10.3 MiB per tile in int32 - 7.4 GiB across a 9^3 grid,
    all resident in the parent while the boundary graph is built. Local ids restart at 1
    in every tile, so they almost always fit uint16 (or uint8), halving or quartering that.
    Comparisons in `build_edges` only test equality/nonzero, so a narrower dtype is
    equivalent; ids are widened back to int64 when packed into the edge arrays.
    """
    idx = tuple(
        (0 if o < 0 else core.shape[d] - 1) if o != 0 else slice(None)
        for d, o in enumerate(offset)
    )
    slab = core[idx]
    if n_local <= np.iinfo(np.uint8).max:
        return slab.astype(np.uint8)
    if n_local <= np.iinfo(np.uint16).max:
        return slab.astype(np.uint16)
    if n_local <= np.iinfo(np.uint32).max:
        return slab.astype(np.uint32)
    return np.array(slab)


def _tile_metadata(
    tile_idx: Tile,
    core_slice: Tuple[slice, ...],
    mask,
    struct: np.ndarray,
    shape: Sequence[int],
    grid_shape: Tile,
    connectivity: int,
    with_properties: bool = False,
    labeler: str = "scipy",
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

    NO HALO: only the tile's own core is read and labelled (see the module docstring) -
    cross-tile merging is entirely the boundary graph's job.
    """
    ndim = len(core_slice)
    core = _read_block(mask, core_slice) > 0
    core, n = _label_block(core, struct, connectivity, labeler)

    # Classify local ids via BOOLEAN PRESENCE masks over 1..n instead of Python sets:
    # `mask[id_array] = True`, `flatnonzero`, `.max()` are all C loops that RELEASE the GIL,
    # so this bookkeeping (the ~13% GIL-bound tail of Phase A) parallelizes across the thread
    # pool - unlike the old `{int(v) for v in np.unique(slab)}` comprehensions, which iterated
    # every id in Python under the GIL. `touching_ids`/`all_ids` are returned as ndarrays.
    boundaries: Dict[Tuple[int, ...], np.ndarray] = {}
    touch = np.zeros(n + 1, dtype=bool)     # touch[id] -> id appears on an inter-tile face
    if n > 0:
        for off in neighbor_offsets(ndim, connectivity):
            neighbor = tuple(t + o for t, o in zip(tile_idx, off))
            if not all(0 <= neighbor[a] < grid_shape[a] for a in range(ndim)):
                continue
            slab = _boundary_slab(core, off, n)
            flat = slab.reshape(-1)
            if flat.size and flat.max() > 0:     # slab carries a foreground piece
                touch[flat] = True
                boundaries[off] = slab
        touch[0] = False                          # background is not a piece

    touching_ids = np.flatnonzero(touch)          # sorted local ids touching a face
    # Labelling the CORE (no halo) means every id 1..n occurs in it by construction, so
    # `all_ids` is exactly 1..n - no scan of the tile needed (with a halo we had to find,
    # and later discard, the "halo-only" components that never reach the core).
    all_ids = np.arange(1, n + 1)
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


class EdgeList:
    """Boundary-graph edges held as two parallel int64 arrays of PACKED piece ids.

    A piece ``(tile_idx, local_id)`` packs to ``tile_ordinal * stride + local_id`` where
    ``tile_ordinal`` is the tile's C-order index in the grid. Keeping the graph in arrays
    (instead of a Python list of nested tuples) is what lets `reconcile` run its union-find
    over plain ints: building 20k+ nested tuples and re-hashing them cost more than the
    graph algorithm itself.

    Still supports ``len()`` and iteration yielding ``((tile_idx, lid), (tile_idx, lid))``
    so the legacy eager path (`tilewise_ccl.legacy`) keeps working unchanged.
    """

    __slots__ = ("a", "b", "grid_shape", "stride")

    def __init__(self, a: np.ndarray, b: np.ndarray, grid_shape: Tile, stride: int) -> None:
        self.a = a
        self.b = b
        self.grid_shape = grid_shape
        self.stride = stride

    def __len__(self) -> int:
        return int(self.a.size)

    def _unpack(self, packed: int) -> Piece:
        ordinal, lid = divmod(int(packed), self.stride)
        return (tuple(int(i) for i in np.unravel_index(ordinal, self.grid_shape)), lid)

    def __iter__(self):
        for pa, pb in zip(self.a.tolist(), self.b.tolist()):
            yield (self._unpack(pa), self._unpack(pb))


def build_edges(
    tile_results_by_idx: Dict[Tile, dict], grid_shape: Tile, connectivity: int,
    release: bool = True,
) -> "EdgeList":
    """Link boundary pieces that touch across a shared tile boundary.

    Considers EVERY neighbor direction admissible under `connectivity` (faces,
    and for connectivity > 1 also edge/corner-diagonal tile neighbors), each
    unordered tile pair once. For a direction Delta, tile A's boundary slab
    facing Delta is compared against neighbor B's slab facing -Delta, at the
    transverse (within-boundary) offsets the connectivity still permits
    (count_nonzero(transverse) <= connectivity - count_nonzero(Delta)). Missing
    the diagonal directions would split objects that cross a tile boundary only
    through a corner/edge.

    `release` (default True) DROPS each cached boundary slab as soon as its comparison is
    done. Every slab takes part in exactly one canonical tile-pair comparison (verified for
    connectivity 1/2/3), so holding them all until the function returns pins the full
    Phase-A slab set for no reason - 7.4 GiB at a 9^3 grid of 672^3 tiles, in the parent
    process, concurrently with Phase B's write buffers. Releasing as we go keeps only the
    live frontier. Pass `release=False` to keep `tile_results_by_idx` intact (the legacy
    eager path re-reads nothing, but callers inspecting `boundaries` afterwards would
    otherwise find them gone).
    """
    ndim = len(grid_shape)
    directions = [off for off in neighbor_offsets(ndim, connectivity) if _is_canonical(off)]
    # Pack (tile, local_id) into one int64 so the graph is two flat arrays rather than a
    # list of nested tuples. stride must exceed every tile's local-id count.
    pack_stride = max((r["n_local"] for r in tile_results_by_idx.values()), default=0) + 1
    strides_c = tuple(int(x) for x in np.cumprod((1,) + tuple(grid_shape)[:0:-1])[::-1])
    edge_a: List[np.ndarray] = []
    edge_b: List[np.ndarray] = []
    # Explicit C order: the `release` invariant below depends on tiles being visited in an
    # order where every canonical neighbour comes later, so don't rely on dict ordering.
    for tile_idx in sorted(tile_results_by_idx):
        result = tile_results_by_idx[tile_idx]
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
                a_ids = fa[both]
                b_ids = fb[both]
                # Deduplicate the (a_lid, b_lid) pairs. `np.unique(stack, axis=0)` does this
                # by SORTING STRUCTURED ROWS, which profiled as 58% of build_edges (the
                # single hottest line: ndarray.sort). Packing each pair into one int64 key
                # and calling the plain 1-D np.unique is ~3.1x faster and yields exactly the
                # same pair set (verified against the 2-D form). `stride` must exceed every
                # b id so the packing stays injective; fall back to the row form in the
                # (unreachable in practice) case where the key could overflow int64.
                key_stride = int(b_ids.max()) + 1
                a64 = a_ids.astype(np.int64, copy=False)
                if key_stride and a64.max() <= (np.iinfo(np.int64).max - key_stride) // key_stride:
                    keys = np.unique(a64 * np.int64(key_stride)
                                     + b_ids.astype(np.int64, copy=False))
                    uniq_a = keys // np.int64(key_stride)
                    uniq_b = keys - uniq_a * np.int64(key_stride)
                else:  # pragma: no cover - needs > ~3e9 labels in a single tile
                    pairs = np.unique(np.stack([a_ids, b_ids], axis=-1), axis=0)
                    uniq_a, uniq_b = pairs[:, 0], pairs[:, 1]
                # Vectorized packing - no per-pair Python tuple building.
                ord_a = sum(int(t) * sc for t, sc in zip(tile_idx, strides_c))
                ord_b = sum(int(t) * sc for t, sc in zip(neighbor, strides_c))
                edge_a.append(uniq_a + np.int64(ord_a * pack_stride))
                edge_b.append(uniq_b + np.int64(ord_b * pack_stride))

        if release:
            # Safe to free THIS tile's slabs now. A canonical offset has its first nonzero
            # component > 0, so every neighbour we compare against is LATER in C order than
            # this tile - no already-processed tile can come back and read these. The only
            # reader of tile T's slabs is T's own pass (its +off slabs) and the pass of the
            # tile at T-off (its -off slabs), which runs EARLIER. Verified for connectivity
            # 1/2/3 on several grids: zero reads of a released slab.
            result["boundaries"] = {}

    a_arr = (np.concatenate(edge_a) if edge_a else np.empty(0, dtype=np.int64))
    b_arr = (np.concatenate(edge_b) if edge_b else np.empty(0, dtype=np.int64))
    return EdgeList(a_arr.astype(np.int64, copy=False), b_arr.astype(np.int64, copy=False),
                    tuple(grid_shape), pack_stride)


def reconcile(
    tile_results: List[dict], edges, n_tiles: int, big_offset: int
) -> Tuple[Dict[Piece, int], int]:
    """Union-find over boundary pieces -> {(tile_idx, local_id): final_id}.

    Runs over PACKED int64 piece ids when handed an `EdgeList` (the fast path): hashing a
    plain int is far cheaper than hashing a nested `((i,j,k), lid)` tuple, and the packed
    form is what `build_edges` already produces. A plain list of tuple pairs (the legacy
    eager path) is still accepted and handled exactly as before. The returned `lut` is a
    dict keyed by the tuple pieces either way, so callers are unaffected.
    """
    if isinstance(edges, EdgeList):
        stride = edges.stride
        grid_shape = edges.grid_shape
        strides_c = tuple(int(x) for x in np.cumprod((1,) + tuple(grid_shape)[:0:-1])[::-1])

        uf = UnionFind()
        for a, b in zip(edges.a.tolist(), edges.b.tolist()):
            uf.union(a, b)

        # Every boundary piece, packed the same way build_edges packed them.
        packed_pieces: List[int] = []
        piece_tuples: List[Piece] = []
        for result in tile_results:
            # `touching_ids` is an ndarray on the core path but a set in tilewise_ccl.legacy -
            # accept either, since legacy shares build_edges/reconcile with this module.
            touching = result["touching_ids"]
            ids = (sorted(int(v) for v in touching)
                   if isinstance(touching, (set, frozenset))
                   else np.asarray(touching).tolist())
            if not ids:
                continue
            tile_idx = result["tile_idx"]
            base = sum(int(t) * sc for t, sc in zip(tile_idx, strides_c)) * stride
            for lid in ids:
                packed_pieces.append(base + lid)
                piece_tuples.append((tile_idx, lid))

        groups: Dict[int, List[int]] = defaultdict(list)
        for i, packed in enumerate(packed_pieces):
            groups[uf.find(packed)].append(i)

        lut: Dict[Piece, int] = {}
        next_id = n_tiles * big_offset + 1  # disjoint from every tile's interior-id range
        for members in groups.values():
            for i in members:
                lut[piece_tuples[i]] = next_id
            next_id += 1
        return lut, len(groups)

    # --- legacy path: a plain sequence of ((tile, lid), (tile, lid)) pairs ---
    uf = UnionFind()
    for a, b in edges:
        uf.union(a, b)

    all_pieces: set = set()
    for result in tile_results:
        for lid in result["touching_ids"]:
            all_pieces.add((result["tile_idx"], int(lid)))   # int(): touching_ids is an ndarray

    groups_t: Dict[Piece, List[Piece]] = defaultdict(list)
    for piece in all_pieces:
        groups_t[uf.find(piece)].append(piece)

    lut = {}
    next_id = n_tiles * big_offset + 1  # disjoint from every tile's interior-id range
    for pieces in groups_t.values():
        for piece in pieces:
            lut[piece] = next_id
        next_id += 1

    return lut, len(groups_t)


# --- Phase A worker plumbing -------------------------------------------------------------
# Module level (not a closure) so a ProcessPoolExecutor can pickle it. The heavy, invariant
# context (mask, struct, ...) is shipped ONCE per worker via the pool initializer rather than
# per task; each task then pickles only its small (tile_idx, core_slice). Lazy masks (zarr /
# dask / dyna DynamicArray) pickle as ~2 KiB references, so no array data is copied.
_WORKER = {}


def _init_worker(mask, struct, shape, grid_shape, connectivity, properties, labeler):
    _WORKER.update(mask=mask, struct=struct, shape=shape, grid_shape=grid_shape,
                   connectivity=connectivity, properties=properties,
                   labeler=labeler)


def _meta_worker(item):
    tile_idx, core_slice = item
    w = _WORKER
    return _tile_metadata(
        tile_idx, core_slice, w["mask"], w["struct"], w["shape"], w["grid_shape"],
        w["connectivity"], w["properties"], w["labeler"],
    )


# --- Phase B block functions -------------------------------------------------------------
# Module level and LOCK-FREE so the lazy output is PICKLABLE: closures over `full_luts` (and,
# previously, a `threading.Lock`) made the returned array impossible to serialize, which broke
# `dask.distributed` and any process-based execution. State is bound with `functools.partial`
# (picklable: module-level func + picklable args). Phase B's per-tile progress print is gone
# with the lock - it could never work from a worker process anyway.

def _relabel_block(block, luts, struct, connectivity, labeler, block_info=None):
    """dask Phase B: block == tile (no halo). Relabel the core, apply the tile's LUT."""
    tile_idx = tuple(block_info[0]["chunk-location"])
    local, _ = _label_block(block > 0, struct, connectivity, labeler)
    return luts[tile_idx][local]


def _relabel_region(block, location, luts, tile_shape, shape, struct, connectivity, labeler,
                    out_dtype):
    """dyna Phase B: a pulled REGION may span several whole tiles (align=tile_shape), so
    decompose it into tiles and relabel each tile's own core (reproducing Phase A's local
    ids exactly), applying that tile's LUT. No halo - see the module docstring."""
    ndim = len(shape)
    starts = [lo for lo, _ in location]
    stops = [hi for _, hi in location]
    out = np.zeros(block.shape, dtype=out_dtype)
    tlo = [starts[a] // tile_shape[a] for a in range(ndim)]
    thi = [(stops[a] - 1) // tile_shape[a] for a in range(ndim)]
    for tidx in itertools.product(*[range(tlo[a], thi[a] + 1) for a in range(ndim)]):
        tcs = [tidx[a] * tile_shape[a] for a in range(ndim)]
        tce = [min((tidx[a] + 1) * tile_shape[a], shape[a]) for a in range(ndim)]
        bsl = tuple(slice(tcs[a] - starts[a], tce[a] - starts[a]) for a in range(ndim))
        local, _ = _label_block(block[bsl] > 0, struct, connectivity, labeler)
        out[bsl] = luts[tidx][local]
    return out


class _DynaAsNumpy:
    """Minimal ndarray-like view over a dyna ``DynamicArray`` that materializes on slice.

    ``da.from_array`` only needs ``shape``/``dtype``/``ndim``/``__getitem__``. A raw
    DynamicArray satisfies those, but its ``__getitem__`` returns another DynamicArray, so
    every dask block would stay lazy and ``ndi.label`` would be handed a non-array. This
    wrapper forces each pulled tile to numpy, which is exactly what the dask path expects.
    """

    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    @property
    def shape(self):
        return tuple(self._arr.shape)

    @property
    def dtype(self):
        return np.dtype(self._arr.dtype)

    @property
    def ndim(self):
        return len(self._arr.shape)

    def __getitem__(self, key):
        return np.asarray(self._arr._read_direct(key))


def _finalize(output_labels, diag, diagnostics, properties, t0):
    """Attach the total-time diagnostic and return either the labeled array or
    ``(array, diag)`` - shared by both Phase-B backends."""
    if diagnostics or properties:
        diag["time_total_s"] = time.perf_counter() - t0
        return output_labels, diag
    return output_labels


def label_array(
    mask,
    tile_shape: Optional[Sequence[int]] = None,
    connectivity: int = 2,
    n_workers: int = 1,
    diagnostics: bool = False,
    properties: bool = False,
    verbose: bool = False,
    backend: str = "auto",
    executor: str = "thread",
    labeler: str = "scipy",
) -> Union[da.Array, Tuple[da.Array, dict]]:
    """Label connected components of a binary N-D mask into a lazy dask array.

    Parameters
    ----------
    mask : zarr.Array | dask.array.Array | np.ndarray
        N-D array, truthy where foreground.
    tile_shape : sequence of int, optional
        Tile size along each axis; must have the same length as `mask.ndim`.
        `None` (the default) derives one from the mask via `default_tile_shape`: a
        ~256 MiB int32 working set over the trailing 3 (spatial) axes, snapped to whole
        storage chunks when the mask exposes `.chunks`, with leading (t/c-like) axes left
        at 1. Execution hyperparameter, independent of storage chunk size but always an
        integer multiple of it when the chunking is known.
    connectivity : int
        Passed to `scipy.ndimage.generate_binary_structure(ndim, connectivity)`.
    n_workers : int
        If > 1, Phase A (metadata pass) runs via a thread pool. Phase B is a
        lazy dask graph - its concurrency is controlled by dask's scheduler
        when the caller computes/writes the result.
    backend : {"auto", "dask", "dyna"}
        Read backend for the tile pulls. ``"auto"`` (default) picks ``"dyna"`` when
        ``mask`` is a ``dyna_zarr`` ``DynamicArray`` and ``"dask"`` otherwise - the mask
        type already determines which path is meant, since the dyna path accepts nothing
        else, so passing this explicitly is only needed to force the dask path for a
        DynamicArray mask. ``"dask"`` reads ``mask`` as a dask/zarr/numpy array.
        ``"dyna"`` expects ``mask`` to be a ``dyna_zarr``
        ``DynamicArray`` (e.g. a lazy threshold like ``io.read(path) > 128``, with
        ``from dyna_zarr import io``):
        Phase A pulls each haloed tile through dyna (memory-bounded, and it FUSES a
        preceding threshold into the tile read - no dask graph/scheduler overhead, ~3x
        faster per tile in practice), and Phase B is a NATIVE dyna ``map_overlap`` (no dask).
        The returned array is then a lazy ``dyna_zarr`` ``DynamicArray`` (``backend="dask"``
        returns a ``dask.array``). Write it with ``dyna_zarr.io.write``: the transform
        carries ``tile_shape`` as its read alignment, and the writer adopts that as its
        region size, so each tile is read and labeled exactly once without the caller
        restating it.
    executor : {"thread", "process"}
        Pool type for Phase A (ignored when `n_workers <= 1`). `"process"` sidesteps the GIL;
        whether that helps depends ENTIRELY on `backend`, because the two differ in how much
        GIL-bound Python work each tile read costs. Measured end to end on 1024^3 uint8 /
        256^3 tiles / 64^3 chunks / 8 workers (best of 2). BOTH backends write the same
        zarr v3 array with the same blosc/zstd-5/bitshuffle codec and produce byte-identical
        output (82.6 MiB on disk in every row); Phase B is timed by that write:

            backend/executor   phase A   phase B    total
            dask / thread        7.0 s    12.8 s   19.7 s
            dask / process       5.1 s    12.7 s   17.8 s
            dyna / thread        2.8 s     8.2 s   11.0 s   <- fastest
            dyna / process       4.4 s     8.1 s   12.5 s

        - `backend="dask"`: the read goes through dask's graph + zarr's Python-level chunk
          assembly, which is GIL-bound, so `"process"` wins Phase A (1.4x).
        - `backend="dyna"`: dyna pulls each tile directly (no dask graph), so there is little
          GIL contention left and the per-tile result pickling costs more than it saves -
          `"thread"` is FASTER (1.6x). Keep the default here.

        NOTE Phase B is 65-75% of the run and `executor` does not affect it, so the BACKEND
        choice dominates: dyna/thread is 1.8x dask/thread overall, while swapping executor
        moves the total by ~10%.
        The mask is shipped to each worker ONCE via the pool initializer, and lazy masks
        (zarr / dask / dyna `DynamicArray`) pickle as ~2 KiB references, so no array data is
        copied. Falls back to threads automatically for an in-memory `np.ndarray` mask (which
        WOULD be copied per worker) and for grids too small to amortize process start-up.
    labeler : {"scipy", "cc3d"}
        Per-tile connected-components kernel. `"scipy"` (default) is `ndi.label`. `"cc3d"`
        (needs `pip install connected-components-3d`, 3-D only) is ~2x faster per call but
        HOLDS THE GIL, so it only pays off with `executor="process"` (with threads it is ~1.5x
        SLOWER overall: 19.6s vs 13.1s). Both emit byte-identical label ids, so this never
        changes the result - it is purely a speed knob.
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
    if backend not in ("auto", "dask", "dyna"):
        raise ValueError(f"backend must be 'auto', 'dask' or 'dyna', got {backend!r}")
    # "auto" (the default): a dyna_zarr DynamicArray is the ONLY input the dyna path
    # accepts, and it is never a valid dask/zarr/numpy input, so the mask type already
    # says which backend is meant - `_with_transform` is the DynamicArray marker the
    # explicit path checks for anyway. Passing backend= is then only needed to force the
    # dask path for a DynamicArray mask (which works: dyna arrays expose __array__).
    if backend == "auto":
        backend = "dyna" if hasattr(mask, "_with_transform") else "dask"
    if executor not in ("thread", "process"):
        raise ValueError(f"executor must be 'thread' or 'process', got {executor!r}")
    if labeler not in ("scipy", "cc3d"):
        raise ValueError(f"labeler must be 'scipy' or 'cc3d', got {labeler!r}")
    if labeler == "cc3d":
        try:
            import cc3d  # noqa: F401
        except ImportError:
            raise ImportError(
                "labeler='cc3d' needs the connected-components-3d package: "
                "pip install connected-components-3d"
            ) from None
    if backend == "dyna":
        # Check the package is importable up front, for the same reason as the mask-type
        # check below: Phase A is expensive and Phase B is where dyna_zarr is actually
        # imported, so without this the failure would land after all that work.
        try:
            import dyna_zarr  # noqa: F401
        except ImportError:
            raise ImportError(
                "backend='dyna' needs the dyna-zarr package: pip install 'tilewise-ccl[dyna]'"
            ) from None
    if backend == "dyna" and not hasattr(mask, "_with_transform"):
        # fail fast (BEFORE the expensive eager Phase A) on a backend/mask mismatch: the dyna
        # Phase B is a native dyna map_overlap, which needs a dyna_zarr DynamicArray.
        raise TypeError(
            f"backend='dyna' needs a dyna_zarr DynamicArray mask, got "
            f"{type(mask).__module__}.{type(mask).__name__}. Open it with dyna_zarr "
            f"(`from dyna_zarr import io; io.read(path)`) and threshold it lazily, e.g. "
            f"`io.read(path) > 128`. A dask/zarr/numpy mask is backend='dask' (which the "
            f"default backend='auto' picks for you)."
        )
    shape = tuple(int(s) for s in mask.shape)
    ndim = len(shape)
    if tile_shape is None:
        # Derive it from the mask: a ~64 MiB working set, grown in whole storage chunks
        # where the mask exposes them. See `default_tile_shape`.
        tile_shape = default_tile_shape(shape, getattr(mask, "chunks", None))
    elif len(tile_shape) != ndim:
        raise ValueError(f"tile_shape has {len(tile_shape)} dims but mask has {ndim}")
    struct = ndi.generate_binary_structure(ndim, connectivity)
    grid_shape = compute_grid_shape(shape, tile_shape)
    n_tiles = int(np.prod(grid_shape))
    big_offset = int(np.prod([t + 2 for t in tile_shape]))  # safe upper bound on n per tile
    tiles = list(iter_tiles(shape, tile_shape))

    # --- Phase A (eager): metadata pass.
    # Runs the module-level `_meta_worker` (picklable) over a thread OR process pool; the
    # invariant context goes to each worker once via the initializer. Progress is printed in
    # the PARENT as results arrive (a worker process can't update the parent's counter). ---
    use_process = executor == "process" and n_workers > 1
    if use_process and isinstance(mask, np.ndarray):
        use_process = False   # an in-memory array would be COPIED into every worker
    if use_process and n_tiles < 2 * n_workers:
        use_process = False   # too few tiles to amortize process start-up

    init_args = (mask, struct, shape, grid_shape, connectivity, properties, labeler)

    t0 = time.perf_counter()
    if verbose:
        print(f"Phase A: computing metadata for {n_tiles} tiles (grid_shape={grid_shape}) "
              f"[{'process' if use_process else 'thread'} pool x{n_workers}, labeler={labeler}]...")
    if n_workers > 1:
        Pool = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        with Pool(max_workers=n_workers, initializer=_init_worker, initargs=init_args) as ex:
            results_iter = ex.map(_meta_worker, tiles)
            tile_results = []
            for result in results_iter:
                tile_results.append(result)
                if verbose:
                    n_interior = len(result["all_ids"]) - len(result["touching_ids"])
                    print(
                        f"  [Phase A {len(tile_results)}/{n_tiles}] tile {result['tile_idx']}: "
                        f"n_local={result['n_local']}, interior={n_interior}, "
                        f"boundary={len(result['touching_ids'])}"
                    )
    else:
        _init_worker(*init_args)
        tile_results = []
        for item in tiles:
            result = _meta_worker(item)
            tile_results.append(result)
            if verbose:
                n_interior = len(result["all_ids"]) - len(result["touching_ids"])
                print(
                    f"  [Phase A {len(tile_results)}/{n_tiles}] tile {result['tile_idx']}: "
                    f"n_local={result['n_local']}, interior={n_interior}, "
                    f"boundary={len(result['touching_ids'])}"
                )
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
    # Every local id 1..n lives in the core (no halo), and touching_ids is a subset of it,
    # so the interior count is just the size difference.
    n_interior_total = sum(len(r["all_ids"]) - len(r["touching_ids"]) for r in tile_results)
    n_final = n_interior_total + n_groups
    out_dtype = np.int32 if n_final <= np.iinfo(np.int32).max else np.int64

    # VECTORIZED per tile: the old form was a Python loop over EVERY local id in every
    # tile (serial, between the two phases) - measured 2s / 16s / 160s for 10k / 100k / 1M
    # ids per tile over a 729-tile grid. Both branches are pure array work:
    #   * interior ids (touch no inter-tile face) take consecutive fresh dense ids, i.e. a
    #     contiguous arange from the running counter;
    #   * boundary ids map through their union-find group, deduplicated in first-appearance
    #     order so the dense numbering is byte-identical to the loop's.
    # Tiles are still visited in the same order, and within a tile ids ascend, so the ids
    # produced are exactly the ones the scalar version produced.
    group_dense: Dict[int, int] = {}  # sparse reconcile id -> dense id (1..n_groups)
    next_interior = n_groups + 1
    full_luts: Dict[Tile, np.ndarray] = {}
    for tile_idx, result in tile_results_by_idx.items():
        n = result["n_local"]
        full_lut = np.zeros(n + 1, dtype=out_dtype)
        if n == 0:
            full_luts[tile_idx] = full_lut
            continue

        touching = result["touching_ids"]
        if touching.size:
            # Boundary ids, ascending (touching_ids is already sorted). Resolve each to its
            # union-find group, then assign dense ids in FIRST-APPEARANCE order across the
            # whole tile sweep - `group_dense` carries over between tiles exactly as before.
            sparse_ids = np.fromiter((lut[(tile_idx, int(l))] for l in touching),
                                     dtype=np.int64, count=touching.size)
            for sp in sparse_ids:
                if sp not in group_dense:
                    group_dense[sp] = len(group_dense) + 1
            full_lut[touching] = np.fromiter(
                (group_dense[sp] for sp in sparse_ids),
                dtype=out_dtype, count=sparse_ids.size)

        # Interior ids = 1..n minus the touching ones -> one contiguous block of fresh ids.
        interior_mask = np.ones(n + 1, dtype=bool)
        interior_mask[0] = False
        interior_mask[touching] = False
        n_interior = int(np.count_nonzero(interior_mask))
        if n_interior:
            full_lut[interior_mask] = np.arange(
                next_interior, next_interior + n_interior, dtype=out_dtype)
            next_interior += n_interior

        full_luts[tile_idx] = full_lut

    n_pass2_tiles = sum(1 for r in tile_results if r["touching_ids"].size)

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

    # --- Phase B (lazy): re-label each tile's CORE + apply its per-tile LUT. No halo, so no
    # overlap machinery and no cross-block reads - each tile is fully independent:
    #   * dask: da.map_blocks with block == tile (chunks=tile_shape); tile index from
    #     block_info['chunk-location'].
    #   * dyna: a native pull-model dyna map_overlap(depth=0) (NO dask). A pulled REGION may
    #     span several whole tiles (align=tile_shape), so the block func decomposes it into
    #     tiles via dyna's block_info `location` and relabels each tile's own core, exactly
    #     reproducing Phase A's local ids.
    if verbose:
        print(
            f"Phase B: lazy graph built ({n_pass2_tiles}/{n_tiles} tiles with "
            f"boundary pieces) - the work happens on .compute()/.to_zarr()/io.write()"
        )

    if backend == "dyna":
        _apply_region = partial(_relabel_region, luts=full_luts,
                                tile_shape=tuple(int(t) for t in tile_shape), shape=shape,
                                struct=struct, connectivity=connectivity, labeler=labeler,
                                out_dtype=out_dtype)

        from dyna_zarr import operations as _dops
        # align=tile_shape: every pulled region is expanded to WHOLE tiles, so Phase B is
        # correct for any write region size / chunking (a sub-tile region just re-reads its
        # tile). `io.write` picks this alignment up automatically as its region size, so
        # each tile is read and labelled exactly once without the caller restating it.
        return _finalize(_dops.map_overlap(mask, _apply_region, depth=0, boundary="constant",
                                           dtype=out_dtype, block_info=True,
                                           align=tuple(int(t) for t in tile_shape)),
                         diag, diagnostics, properties, t0)

    # --- dask backend ---
    # `rechunk` alone is not proof of a dask array: a dyna_zarr DynamicArray also exposes
    # one (as a documented no-op, since the pull model is chunk-invariant). Feeding that
    # straight to map_blocks yields a graph whose blocks never get `block_info`, and Phase B
    # then dies with `KeyError: 0` on compute - long after this call returned. Require a
    # real dask array, and wrap anything else.
    if isinstance(mask, da.Array):
        mask_da = mask.rechunk(tile_shape)
    elif hasattr(mask, "_with_transform"):
        # A DynamicArray on the DASK path (backend="dask" forced, or auto with a dyna mask
        # the caller wants as a dask array). `da.from_array` would keep each block lazy -
        # slicing a DynamicArray returns another DynamicArray, not ndarray - and ndi.label
        # then fails on a non-array block. Materialize each tile as it is pulled.
        mask_da = da.from_array(_DynaAsNumpy(mask), chunks=tile_shape)
    else:
        mask_da = da.from_array(mask, chunks=tile_shape)

    output_labels = da.map_blocks(
        partial(_relabel_block, luts=full_luts, struct=struct, connectivity=connectivity,
                labeler=labeler),
        mask_da,
        dtype=out_dtype,
        meta=np.array((), dtype=out_dtype),
    )
    return _finalize(output_labels, diag, diagnostics, properties, t0)
