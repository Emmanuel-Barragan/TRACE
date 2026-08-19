# This module handles recovery of metrics like stress
from fe.quad4 import isoparametric_calc_quad4
import numpy as np 


def nodal_stress_quad4(nat_coords, thickness, E_matrix, u_element):
    """ Calculates the nodal stresses for a quad 4 element

    Input: nat_coords (list format)
           thickness (float)
           E_matrix, plains stress elastic modulii (3x3 matrix)
           u_element, the 8 displacement components for a 4quad
    Output: the nodal stresses , stress_element
    """
    nodal_iso_coords_list = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

    stress_element = np.zeros((4,3))

    for node in range(4):
        iso_coords = nodal_iso_coords_list[node]

        N, dN_dx, dN_dy, detJ = isoparametric_calc_quad4(nat_coords, iso_coords)

        B = np.zeros((3, 8))
        
        # ::2 means every other element.
        B[0, ::2]  = dN_dx  
        B[1, 1::2] = dN_dy
        B[2, ::2]  = dN_dy; B[2, 1::2] = dN_dx
        
        stress_element[node, :] = E_matrix @ (B @ u_element)

    return stress_element





