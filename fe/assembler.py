# This module handles master stiffness assembly.
from fe.quad4 import stiffness_matrix_quad4, E_matrix_generation
import numpy as np


def assembler_master_stiffness(list_coords, element_node_list, thickness, E_modulus, nu):
    """ Assembles global stiffness matrix for 2D QUAD4 mesh.

    Input: list_coords [[x1.y1],[x2,y2], etc]
           element_node_list (connectivity, list of nodes for every element, 0 based index)
           thickness (float)
           E_modulus (float)
           nu (float)

    Output:  master stiffness K
    """
    list_coords = np.asarray(list_coords)
    number_nodes = len(list_coords)

    K_global = np.zeros((2*number_nodes, 2*number_nodes))
    
    E_matrix = E_matrix_generation(E_modulus, nu) 

    for e in element_node_list:
        nat_coords = list_coords[e] 

        element_dofs = np.array([
            [2*k, 2*k+1] for k in e]).flatten()
        
        Ke = stiffness_matrix_quad4(nat_coords, thickness, E_matrix)

        number_DoF = Ke.shape[0] # always 8 (4 nodes x 2 DoF)

        for i in range(number_DoF):
            for j in range(i, number_DoF):
                K_global[element_dofs[i], element_dofs[j]] += Ke[i, j]
                K_global[element_dofs[j], element_dofs[i]] = K_global[element_dofs[i], element_dofs[j]]

    return K_global



