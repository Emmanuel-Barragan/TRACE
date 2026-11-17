# This module handles recovery of metrics like stress
from fe.quad4 import isoparametric_calc_quad4, kinematic_relation, E_matrix_generation
import numpy as np 

def nodal_stress_quad4(nat_coords, E_matrix, u_element):
    """ Calculates the nodal stresses for a quad 4 element

    Input: nat_coords (list format)
           E_matrix, plains stress elastic modulii (3x3 matrix)
           u_element, the 8 displacement components for a 4quad
    Output: the nodal stresses , stress_element
    """
    nodal_iso_coords_list = [[-1, -1], [1, -1], [1, 1], [-1, 1]]

    stress_element = np.zeros((4,3))

    for node in range(4):
        iso_coords = nodal_iso_coords_list[node]

        N, dN_dx, dN_dy, detJ = isoparametric_calc_quad4(nat_coords, iso_coords)

        B = kinematic_relation(dN_dx, dN_dy)
        
        stress_element[node, :] = E_matrix @ (B @ u_element)

    return stress_element


def total_nodal_stresses(list_coords, element_node_list, E_modulus, nu, u_vector):
    """ Generates the stress profile at every node in the system

    Input: list_coords [[x1.y1],[x2,y2], etc]
           element_node_list (connectivity, list of nodes for every element, 0 based index)
           E_modulus (float)
           nu (float)
           u_vector, displacement u_vector

    Output: total_stress, a number_nodes x 3 matrix
    """
    list_coords = np.asarray(list_coords)
    number_nodes = len(list_coords)
    E_matrix = E_matrix_generation(E_modulus, nu) 

    total_stress = np.zeros((number_nodes, 3)) # running sum of stress per node
    nodal_count = np.zeros((number_nodes))  #elements connected to every node

    for e in element_node_list:
        nat_coords = list_coords[e] 

        element_dofs = np.array([[2*k, 2*k+1] for k in e]).flatten()

        u_element = u_vector[element_dofs]

        stress_element = nodal_stress_quad4(nat_coords, E_matrix, u_element)

        for local_i, global_k in enumerate(e):
            total_stress[global_k] += stress_element[local_i]
            nodal_count[global_k] += 1

    return total_stress / nodal_count[:, None] #numpy broadcast division, for each row, divides corresponding stresses by nodal count.





