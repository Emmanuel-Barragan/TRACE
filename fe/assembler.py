# This module handles master stiffness assembly.
from fe.quad4 import stiffness_matrix_quad4
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
    
    number_nodes = len(list_coords)
    number_elements = len(element_node_list)

    K_global = np.zeros((2*number_nodes, 2*number_nodes))

    E_matrix = E_modulus / (1 - nu ** 2) * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1 - nu) / 2],
    ])

    for e in range(number_elements):
        element_nodes = element_node_list[e]
        
        freedom_table = []
        nat_coords = []

        for i in element_nodes:
            freedom_table.extend([2*i , 2*i+1])
            nat_coords.append(list_coords[i])
        
        Ke = stiffness_matrix_quad4(nat_coords, thickness, E_matrix)

        number_DoF = Ke.shape[0] # always 8 (4 nodes x 2 DoF)

        for i in range(number_DoF):
            global_i = freedom_table[i]
            for j in range(i, number_DoF):
                global_j = freedom_table[j]
                K_global[global_i, global_j] += Ke[i, j]
                K_global[global_j, global_i] = K_global[global_i, global_j]

    return K_global


