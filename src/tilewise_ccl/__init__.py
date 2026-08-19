"""tilewise-ccl: fast tile-wise connected-components labeling for large N-D arrays.

`label_array` returns a lazy `dask.array.Array` of connected-component labels
for any large N-D binary mask (numpy / zarr / dask input), computed via a
tile-wise local labeling + global boundary-piece union-find graph. It is
storage- and domain-agnostic (no OME-Zarr dependency) and composes with
downstream dask operations.

`label_array_legacy` is the earlier eager implementation (writes into a
pre-allocated output buffer); kept for comparison/benchmarking.
"""

__version__ = "0.0.4"
__author__ = "Bugra Oezdemir"

from tilewise_ccl.core import label_array
from tilewise_ccl.legacy import label_array_legacy

__all__ = ["label_array", "label_array_legacy"]
