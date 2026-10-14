# -*- coding: utf-8 -*-
"""Steady 2D seepage under a sheet-pile wall, solved with CALFEM.

The domain is a 100 x 10 block of saturated soil with a thin slot (1 wide,
5 deep) cut down from the middle of the top edge. The slot represents an
impermeable wall. The hydraulic head (potential) is fixed at 10 on the
top-left surface and at 0 on the top-right surface; every other edge,
including the slot faces, is a no-flow boundary.

The script solves the Laplace equation

    -div(D grad(phi)) = 0

with 4-node isoparametric quadrilateral finite elements, then computes the
Darcy flux q = -D grad(phi) in each element. It plots the geometry, the flux
magnitude per element and the nodal potential.

The same equation describes steady heat conduction (phi = temperature) or
electric conduction (phi = voltage); only the interpretation changes.

Assumptions: steady state, homogeneous isotropic material (D = identity),
unit thickness, no internal sources, dimensionless units.

Behaviour is identical to ex1_original.py; only comments were added.
"""

import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu

# --- Problem parameters ----------------------------------------------------

w = 100.0  # domain width
h = 10.0  # domain height
t = 1.0  # slot (wall) width
d = h / 2  # slot (wall) depth, measured down from the top edge

# Constitutive matrix [[kxx, kxy], [kyx, kyy]]: isotropic, unit permeability.
D = np.identity(2, "float")

# Element properties [thickness, ir]. ir is the Gauss integration rule:
# ir = 1 means a single integration point per element. Note that one point
# under-integrates the bilinear quad (its element matrix gets a spurious
# zero-energy "hourglass" mode); ir = 2 is the usual choice.
ep = [1.0, 1]

# --- Geometry --------------------------------------------------------------

g = cfg.Geometry()

# Points are referenced by their 0-based creation index in the calls below.
g.point([0, 0])  # 0: bottom-left
g.point([w, 0])  # 1: bottom-right
g.point([w, h])  # 2: top-right
g.point([w - w / 2 + t / 2, h])  # 3: slot, top-right corner
g.point([w - w / 2 + t / 2, h - d])  # 4: slot, bottom-right corner
g.point([w - w / 2 - t / 2, h - d])  # 5: slot, bottom-left corner
g.point([w - w / 2 - t / 2, h])  # 6: slot, top-left corner
g.point([0, h])  # 7: top-left

# Marker IDs label boundary curves so boundary conditions can be applied to
# them later. The names are misleading: left_side is placed on the top-RIGHT
# edge and right_side on the top-LEFT edge (see the curves below).
left_side = 80
right_side = 90

# Straight edges (a two-point spline is a line), traced counter-clockwise.
g.spline([0, 1])  # bottom edge
g.spline([1, 2])  # right edge
g.spline([2, 3], marker=left_side)  # top edge, right of the slot
g.spline([3, 4])  # slot, right face
g.spline([4, 5])  # slot, bottom face
g.spline([5, 6])  # slot, left face
g.spline([6, 7], marker=right_side)  # top edge, left of the slot
g.spline([7, 0])  # left edge

# One surface bounded by all eight curves (indices into the curve list).
g.surface([0, 1, 2, 3, 4, 5, 6, 7])

# --- Mesh ------------------------------------------------------------------

el_type = 3  # Gmsh element type 3: 4-node quadrilateral
dofs_per_node = 1  # scalar field: one unknown (the potential) per node

mesh = cfm.GmshMesh(g)
mesh.el_size_factor = 1.0  # relative element size; smaller gives a finer mesh
mesh.el_type = el_type
mesh.dofs_per_node = dofs_per_node

# coords:         (n_nodes, 2) node coordinates
# edof:           (n_elements, 4) global DOF numbers of each element (1-based)
# dofs:           (n_nodes, 1) DOF number of each node
# bdofs:          dict mapping a boundary marker to the DOFs on that boundary
# elementmarkers: surface marker of each element (unused here)
coords, edof, dofs, bdofs, elementmarkers = mesh.create()

# --- Assemble the global system -------------------------------------------

nDofs = np.size(dofs)

# ex, ey: (n_elements, 4) x and y coordinates of each element's corner nodes.
ex, ey = cfc.coordxtr(edof, coords, dofs)

K = np.zeros([nDofs, nDofs])

for eltopo, elx, ely in zip(edof, ex, ey):
    # 4x4 element conductivity matrix Ke = integral of B^T D B * thickness.
    Ke = cfc.flw2i4e(elx, ely, ep, D)
    # Add Ke into K at the rows/columns given by this element's DOFs.
    cfc.assem(eltopo, K, Ke)

# No internal sources or sinks, so the load vector is zero.
f = np.zeros([nDofs, 1])

# --- Boundary conditions ---------------------------------------------------

# bc holds the constrained DOF numbers, bcVal their prescribed values.
bc = np.array([], int)
bcVal = np.array([], int)

# Prescribed potential (Dirichlet conditions). Edges without a condition get
# zero normal flux automatically (natural Neumann condition), so the bottom,
# the two vertical sides and the slot faces are impermeable.
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, left_side, 0.0)  # top-right: phi = 0
bc, bcVal = cfu.applybc(bdofs, bc, bcVal, right_side, 10.0)  # top-left: phi = 10

# --- Solve -----------------------------------------------------------------

# Solve K a = f with the prescribed values eliminated.
# a: nodal potentials. r: boundary flows at the constrained DOFs (inflow is
# positive). r sums to zero, a useful mass-balance check, but is unused here.
a, r = cfc.solveq(K, f, bc, bcVal)

# ed: (n_elements, 4) nodal potentials of each element, picked out of a.
ed = cfc.extract_eldisp(edof, a)

# --- Post-process: flux per element ---------------------------------------

# Despite the name, this holds the flux magnitude |q| at the first Gauss
# point of each element, not a maximum. With ir = 1 that point is the element
# centre; with ir > 1 only the first of the ir*ir points would be used.
maxFlow = []

for i in range(edof.shape[0]):
    # es: flux q = -D grad(phi) at each Gauss point, rows [qx, qy]
    # et: gradient grad(phi) at each Gauss point
    # eci: Gauss point coordinates
    es, et, eci = cfc.flw2i4s(ex[i, :], ey[i, :], ep, D, ed[i, :])
    maxFlow.append(np.sqrt(pow(es[0, 0], 2) + pow(es[0, 1], 2)))

# --- Plot ------------------------------------------------------------------

cfv.figure()
cfv.draw_geometry(g, title="Geometry")

cfv.figure()
cfv.draw_element_values(
    maxFlow, coords, edof, dofs_per_node, el_type, None, title="Max flows"
)

cfv.figure()
cfv.draw_nodal_values(a, coords, edof, dofs_per_node=dofs_per_node, el_type=el_type)

cfv.showAndWait()
