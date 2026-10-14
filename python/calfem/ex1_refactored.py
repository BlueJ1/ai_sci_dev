# -*- coding: utf-8 -*-
"""Steady 2D seepage under a sheet-pile wall, solved with CALFEM.

The domain is a rectangular block of saturated soil with a thin slot cut down
from the middle of the top edge; the slot represents an impermeable wall. The
potential (hydraulic head) is fixed on the two top surfaces either side of the
wall, and every other edge is a no-flow boundary.

The script solves -div(D grad(phi)) = 0 with 4-node quadrilateral elements,
computes the flux q = -D grad(phi) in each element, and plots the geometry,
the flux magnitude and the potential.

Refactored from ex1_original.py; the results are identical.
"""

import numpy as np
import calfem.core as cfc
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import calfem.utils as cfu

# Domain and wall dimensions.
DOMAIN_WIDTH = 100.0
DOMAIN_HEIGHT = 10.0
WALL_THICKNESS = 1.0
WALL_DEPTH = DOMAIN_HEIGHT / 2

# Material and element settings.
CONDUCTIVITY = np.identity(2, "float")  # isotropic, unit permeability
ELEMENT_THICKNESS = 1.0
INTEGRATION_RULE = 1  # Gauss points per direction; 2 is the usual choice
ELEMENT_TYPE = 3  # Gmsh type 3: 4-node quadrilateral
DOFS_PER_NODE = 1  # one unknown (the potential) per node
ELEMENT_SIZE_FACTOR = 1.0

# Boundary markers. In ex1_original.py these were called left_side and
# right_side, but they sit on the top-right and top-left edges respectively.
TOP_RIGHT_MARKER = 80
TOP_LEFT_MARKER = 90

# Prescribed potentials on the marked boundaries.
TOP_RIGHT_POTENTIAL = 0.0
TOP_LEFT_POTENTIAL = 10.0


def create_geometry(width, height, wall_thickness, wall_depth):
    """Build the outline of a rectangle with a slot cut down from its top edge.

    The slot is centred horizontally. The two top edges either side of it
    get the boundary markers TOP_LEFT_MARKER and TOP_RIGHT_MARKER.

    Args:
        width: Width of the domain.
        height: Height of the domain.
        wall_thickness: Width of the slot.
        wall_depth: Depth of the slot, measured down from the top edge.

    Returns:
        A calfem.geometry.Geometry with one surface.
    """
    wall_left = width / 2 - wall_thickness / 2
    wall_right = width / 2 + wall_thickness / 2
    wall_bottom = height - wall_depth

    g = cfg.Geometry()

    # Corner points, counter-clockwise from the bottom-left.
    g.point([0, 0])  # 0
    g.point([width, 0])  # 1
    g.point([width, height])  # 2
    g.point([wall_right, height])  # 3
    g.point([wall_right, wall_bottom])  # 4
    g.point([wall_left, wall_bottom])  # 5
    g.point([wall_left, height])  # 6
    g.point([0, height])  # 7

    g.spline([0, 1])  # bottom
    g.spline([1, 2])  # right side
    g.spline([2, 3], marker=TOP_RIGHT_MARKER)  # top, right of the wall
    g.spline([3, 4])  # wall, right face
    g.spline([4, 5])  # wall, bottom face
    g.spline([5, 6])  # wall, left face
    g.spline([6, 7], marker=TOP_LEFT_MARKER)  # top, left of the wall
    g.spline([7, 0])  # left side

    g.surface([0, 1, 2, 3, 4, 5, 6, 7])
    return g


def create_mesh(geometry):
    """Mesh the geometry with Gmsh.

    Args:
        geometry: The calfem.geometry.Geometry to mesh.

    Returns:
        coords: (n_nodes, 2) node coordinates.
        edof: (n_elements, 4) 1-based DOF numbers of each element.
        dofs: (n_nodes, 1) DOF number of each node.
        bdofs: Dict mapping each boundary marker to the DOFs on it.
    """
    mesh = cfm.GmshMesh(geometry)
    mesh.el_size_factor = ELEMENT_SIZE_FACTOR
    mesh.el_type = ELEMENT_TYPE
    mesh.dofs_per_node = DOFS_PER_NODE

    coords, edof, dofs, bdofs, _element_markers = mesh.create()
    return coords, edof, dofs, bdofs


def assemble_conductivity_matrix(edof, ex, ey, element_props, D, n_dofs):
    """Assemble the global conductivity ("stiffness") matrix.

    Args:
        edof: (n_elements, 4) DOF numbers of each element.
        ex, ey: (n_elements, 4) corner coordinates of each element.
        element_props: [thickness, integration_rule] for flw2i4e.
        D: (2, 2) conductivity matrix.
        n_dofs: Total number of DOFs.

    Returns:
        K: (n_dofs, n_dofs) global matrix.
    """
    K = np.zeros([n_dofs, n_dofs])
    for element_dofs, elx, ely in zip(edof, ex, ey):
        Ke = cfc.flw2i4e(elx, ely, element_props, D)
        cfc.assem(element_dofs, K, Ke)
    return K


def apply_boundary_conditions(bdofs):
    """Collect the prescribed potentials on the two top edges.

    Edges without a prescribed value are no-flow boundaries; that follows
    from the weak form and needs no code.

    Args:
        bdofs: Dict mapping boundary markers to their DOFs.

    Returns:
        bc: Constrained DOF numbers.
        bc_values: Prescribed potential at each constrained DOF.
    """
    bc = np.array([], int)
    bc_values = np.array([], int)
    bc, bc_values = cfu.applybc(bdofs, bc, bc_values, TOP_RIGHT_MARKER, TOP_RIGHT_POTENTIAL)
    bc, bc_values = cfu.applybc(bdofs, bc, bc_values, TOP_LEFT_MARKER, TOP_LEFT_POTENTIAL)
    return bc, bc_values


def compute_flux_magnitudes(ex, ey, element_props, D, element_potentials):
    """Compute the flux magnitude |q| in each element.

    The flux q = -D grad(phi) is evaluated by flw2i4s at the element's Gauss
    points; only the first point is used. With a 1-point rule that is the
    element centre.

    Args:
        ex, ey: (n_elements, 4) corner coordinates of each element.
        element_props: [thickness, integration_rule].
        D: (2, 2) conductivity matrix.
        element_potentials: (n_elements, 4) nodal potentials of each element.

    Returns:
        List of flux magnitudes, one per element.
    """
    flux_magnitudes = []
    for elx, ely, ed in zip(ex, ey, element_potentials):
        es, _gradients, _gauss_coords = cfc.flw2i4s(elx, ely, element_props, D, ed)
        qx, qy = es[0, 0], es[0, 1]
        flux_magnitudes.append(np.sqrt(qx**2 + qy**2))
    return flux_magnitudes


def plot_results(geometry, flux_magnitudes, potentials, coords, edof):
    """Show the geometry, element flux magnitudes and nodal potentials."""
    cfv.figure()
    cfv.draw_geometry(geometry, title="Geometry")

    cfv.figure()
    cfv.draw_element_values(
        flux_magnitudes, coords, edof, DOFS_PER_NODE, ELEMENT_TYPE, None, title="Max flows"
    )

    cfv.figure()
    cfv.draw_nodal_values(
        potentials, coords, edof, dofs_per_node=DOFS_PER_NODE, el_type=ELEMENT_TYPE
    )

    cfv.showAndWait()


def main():
    element_props = [ELEMENT_THICKNESS, INTEGRATION_RULE]

    geometry = create_geometry(DOMAIN_WIDTH, DOMAIN_HEIGHT, WALL_THICKNESS, WALL_DEPTH)
    coords, edof, dofs, bdofs = create_mesh(geometry)

    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coords, dofs)

    K = assemble_conductivity_matrix(edof, ex, ey, element_props, CONDUCTIVITY, n_dofs)
    f = np.zeros([n_dofs, 1])  # no internal sources
    bc, bc_values = apply_boundary_conditions(bdofs)

    potentials, _boundary_flows = cfc.solveq(K, f, bc, bc_values)

    element_potentials = cfc.extract_eldisp(edof, potentials)
    flux_magnitudes = compute_flux_magnitudes(ex, ey, element_props, CONDUCTIVITY, element_potentials)

    plot_results(geometry, flux_magnitudes, potentials, coords, edof)


if __name__ == "__main__":
    main()
