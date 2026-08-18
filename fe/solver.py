# This modules handles actual solution generation
import numpy as np 
from fe.boundary import modified_stiffness_matrix, modified_force_vector

def solver(K_global, f_vector, prescribed_DoF):
    """ Applies BC and solves for displacement

    Input: K_global (ndarray)
           f_vector (ndarray)
           prescribed_DoF (list of DoF, 0-based)

    Output: u displacement vector 
    """

    K_modified = modified_stiffness_matrix(K_global, prescribed_DoF)
    f_modified = modified_force_vector(f_vector, prescribed_DoF)
    
    u_vector = np.linalg.solve(K_modified, f_modified)

    return u_vector




