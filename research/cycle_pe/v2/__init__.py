"""Cycle V2: matched sparse DFS-cycle SE and SE-plus-relative-PE experiments.

The original cycle-set summary experiment remains unchanged in the parent
package. Signed sparse DFS cycles certify the full incidence left nullspace.
SE summarizes their membership; PE adds a cyclic-relative cosine-kernel
residual from actual ordered cycle positions, without extra parameters, QR,
SVD, dense projectors or edge-pair matrices. Both preserve the same deep
residual molecular backbone and depend on the selected DFS tree, not on
cycle ordering, origin or traversal direction.
"""
