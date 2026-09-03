"""Cycle PE v2: coordinate-free incidence-cycle-space projector PE.

The original cycle-set summary experiment remains unchanged in the parent
package. The production backend caches thin-Q coordinates; the optional DFS
backend may cache raw fundamental cycles but orthonormalizes them before any
learned layer. Both paths use the same intrinsic projector kernel and deep
residual molecular GNN.
"""
